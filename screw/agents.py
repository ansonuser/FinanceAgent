from typing import Union, Dict, Optional, List
from utils.utils import call_api, extract_json, a_call_api
from schema.data import RevenueData
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential_jitter,
    stop_after_attempt
)

from json import JSONDecodeError

retryable_exceptions = (
    JSONDecodeError,
    TypeError,
    ValueError
)
from textwrap import dedent


def on_retry_error(retry_state):
    # This is called when retries are exhausted
    print("All retries failed — continuing gracefully.")
    return {}  # Fallback result


class Extractor:
    """
    A wrapper to extract data from financial
    """
    def __init__(self, sys_prompt: str = None, base_url: str = None, model_name: str = "qwen2.5-coder:14b"):
        self.session = None
        if sys_prompt:
            self._sys_prompt = sys_prompt
        else:
            self.set_sys()
        if base_url:
            self._base_url = base_url.rstrip('/') 
        else:
            self._base_url = "http://localhost:11434/api/chat"

        self._model_name = model_name
        self._isollama = True
    
    def set_ollama(self, state: bool):
        self._isollama = state
    
    def update_source(self, model_name : str, base_url : str):
        self._model_name = model_name
        self._base_url = base_url

    def get_source(self):
        return (self._model_name, self._base_url)

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=3),
        retry=retry_if_exception_type(retryable_exceptions),
        stop=stop_after_attempt(3),
        retry_error_callback=on_retry_error
    )
    async def a_generate(self, prompts: List[str], schema: Optional[RevenueData] = None, use_sys : bool = True) -> Union[str, Dict]:
        """Asynchronously send prompts to the model and return structured or raw results."""
        # ---- Build message list ----
        if use_sys and prompts:
            if len(prompts) >= 1:
                messages = [
                    {"role": "user", "content": self._sys_prompt + '\n\n' + prompts[0]["content"]},
                ]
                messages += prompts[1:]
        else:
            messages = prompts


        # ---- Async API call ----
        json_res = await a_call_api(self.session, messages, self._base_url, model = self._model_name, is_ollama = self._isollama)
        

        if schema:
            if "currency" in json_res and json_res["currency"] == "":
                json_res["currency"] = "USD"
            return schema(**json_res)

        return json_res


    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=3),
        retry=retry_if_exception_type(retryable_exceptions),
        stop=stop_after_attempt(3),
        retry_error_callback=on_retry_error
    )
    def generate(self, prompts: List[str], schema: Optional[RevenueData] = None, use_sys : bool = True) -> Union[str, Dict]:
        if use_sys and prompts:
            if len(prompts) >= 1:
                messages = [
                    {"role" : "user", "content" : self._sys_prompt + '\n\n' + prompts[0]["content"]},
                ] 
                messages += prompts[1:]
        else:
            messages = prompts
 
        res = call_api(messages, self._base_url, self._model_name)
        
        json_res = extract_json(res)
        if schema:
            if "currency" not in json_res:
                json_res["currency"] = "USD"
            return schema(**json_res)
        return json_res
    
    def set_sys(self, mode = 'Q'):
        self._sys_prompt = self.get_sys(mode = mode)

    @staticmethod
    def get_sys(mode ='Q'):
        if mode == 'Q':
            prompt = """
            You are an expert financial report parser. Your task is to extract revenue/sales by product segment from the given text.
            
            ## Rule for Disambiguation in 10-Q Tables
            * Header context: Use the preceding phrase (e.g. “Three Months Ended ,”) to bind each year.
            * Canonical form: Combine header + year (e.g."Three Months Ended June 30 2025" and "Three Months Ended June 30 2024".)        
            ### Normalization logic:
            * Among all periods shown (Three Months, Six Months, Nine Months, Twelve Months, etc.), always select the shortest period that represents a fiscal Quarter (usually 'Three Months Ended', 'Quarter Ended', or equivalent)
            * If the table contains only historical years, treat the latest available year as the Current Quarter for extraction purposes.   

            Follow these rules carefully:

            1. Target Information
            * Extract revenue by product segment for the most recent Quarter(Three Months) available in the input table (the latest year shown).
            * Extract all revenue line items by business segment or product category, e.g. “Payments and other fees”, “Cloud”, “Hardware”, etc.
            * Exclude all percentage signs, labels like “%”, and any percentage-based data.
            * Include their values, currencies, and periods.
            * Include total revenue, if available.
            * Reject ratio-based or percentage-mostly tables.
            * Ignore unrelated metrics (e.g., costs, compensation, R&D, etc.).

            2. Normalization
            * Normalize numbers by removing “$”, commas, and parentheses.
            * Interpret “(5)” as –5.
            * Convert scale (e.g. "in millions" to "millions") appropriately if mentioned.
            * Maintain consistent period labels.(See Rule for Disambiguation in 10-Q Tables)

            3. Output Format
            Return only a VALID JSON object, the commentary must go in the reasoning field.
            Leave the value empty if not value found.
            Follow exactly this schema:

            {
                "period": "2025-06-30",
                "currency": "USD",
                "scale": "thousands",
                "product_segments":
                    {
                    "Mobile" : 1020,
                    "Service": 112
                    },
                "total_revenue" : 1132,
                "reasoning": "<Describe how you find the answer.>"
            }

            4. Error Handling
            * Ensure the data is from the most recent quarter available in the input table
            * If data is partial or ambiguous, include only what is certain.
            * If multiple tables exist, merge by matching periods or headers.
            * Do not include narrative paragraphs unless they explicitly quantify segment revenue.

            5. Rules
            * Do not caculate. Output number must be from input.
            * You Must follow the output format.
            * All information should comes from the following input.
            * Pay attention on `scale`, especially not include percentage-based data.
            * Before finalizing, recheck that all extracted values belong to the “Three Months” or equivalent quarterly column, not to any cumulative period.

            Now analyze the following input:
            """
        else:
            prompt = """
            You are an expert financial report parser. Your task is to extract revenue/sales by product segment from the given text.
            
            ## Rule for Disambiguation in 10-K Tables
            * Header context: Use the preceding phrase (e.g. “Year Ended December 31,” or “Fiscal Year Ended,”) to bind each year.
            * Canonical form: Combine header + year (e.g. "Year Ended December 31 2024" and "Year Ended December 31 2023".)
            ### Normalization logic:
            * Among all periods shown (Three Months, Nine Months, Twelve Months, etc.), always select the **longest annual period** that represents the full fiscal year (usually “Year Ended”, “Twelve Months Ended”, or “Fiscal Year Ended”).
            * If multiple years are presented, use the most recent fiscal year as the "Current Year."
            * If both fiscal and calendar years appear, prefer the fiscal definition explicitly stated in the table.
            * Ignore quarterly or interim data if mixed within an annual table.

            Follow these rules carefully:

            1. Target Information
            * Extract revenue by product segment for the most recent **fiscal year** available in the input table.
            * Extract all revenue line items by business segment or product category, e.g., “Cloud,” “Hardware,” “Subscription services,” etc.
            * Exclude all percentage signs, labels like “%,” and any percentage-based data.
            * Include their values, currencies, and periods.
            * Include total revenue, if available.
            * Reject ratio-based or percentage-mostly tables.
            * Ignore unrelated metrics (e.g., cost of revenue, R&D, operating income, etc.).

            2. Normalization
            * Normalize numbers by removing “$”, commas, and parentheses.
            * Interpret “(5)” as –5.
            * Convert scale (e.g., "in millions" → "millions") appropriately if mentioned.
            * Maintain consistent period labels. (See Rule for Disambiguation in 10-K Tables.)

            3. Output Format
            Return only a VALID one JSON object, not a list. The commentary must go in the reasoning field.
            Leave the value empty if no value is found. 

            Follow exactly this schema:

            {
                "period": "2024-12-30",
                "currency": "USD",
                "scale": "millions",
                "product_segments":
                    {
                    "Cloud Services" : 108520,
                    "Consumer Devices": 45800,
                    "Advertising": 39210
                    },
                "total_revenue" : 193530,
                "reasoning": "<Describe how you find the answer.>"
            }

            4. Error Handling
            * Ensure the data corresponds to the **most recent fiscal year** available.
            * If multiple tables exist, merge by matching “Year Ended” headers or consistent fiscal labels.
            * If data is partial or ambiguous, include only what is certain.
            * Do not include narrative paragraphs unless they explicitly quantify segment revenue.

            5. Rules
            * Do not calculate or infer missing values — only extract explicit numbers from the input.
            * You Must follow the output format exactly.
            * All information must come from the given input.
            * Pay special attention to the `scale`, ensuring no percentage-based data is included.

            Now analyze the following input:
        """
        return dedent(prompt).strip()
        

    @staticmethod
    def send_validation(company_name : str, mode : str = "Q")->str:
        if mode == "Q":
            period = "quarterly"
        else:
            period = "yearly"


        return dedent(
        f"""
        Identify the correct {period} revenue segment by product for {company_name}
        Think in the following process:
        1. Determine how {company_name} earned revenue using primary sources
        2. List {company_name}’s key products
        3. It eithers all keys in segments are valid or invalid.
        4. If multiple candidates exist, choose the one most directly representing revenue by product segment. 
        5. Output a single JSON object only, not a list.
        Internally analyze (without showing reasoning) and verify the correct answer and output it as JSON format at the end.
        Don't not frabricate the numbers.

        Output Example:
        {{
                "period": "1994-06-30",
                "currency": "USD",
                "scale": "thousands",
                "product_segments":
                    {{
                    "Mobile" : 1020,
                    "Service": 112
                    }},
                "total_revenue" : 1132,
                "reasoning": "<Describe how you validate the answer.>"
        }}

        User Input:

        """).strip()
    
    @staticmethod
    def send_refinement(mode = "Q"):
        if mode == 'Q':
            return dedent("""
            You are given a document chunk and an answer derived from it.
            Your job is to internally verify the correctness of each field based on the document and output only the validated structured result.
            Always check missing products or business segemnts if any!!!  
            
            ## Quarter Disambiguation Rule
            In 10-Q filings, tables often present both quarterly and cumulative results.

            - Identify the data that reflects a single fiscal quarter.  
            Quarterly results are typically described with wording such as  
            “for the quarter ended” or “for the three-month period ended.”

            - Distinguish these from cumulative spans that combine multiple quarters,  
            often labeled as “six months ended,” “nine months ended,” or “year-to-date.”  
            These represent totals across several quarters and should not be used when focusing on one quarter’s results.

            - When more than one quarterly period appears (for example, current and prior year),  
            select the one associated with the **later fiscal period** in the table.

            - Always prefer the column that reports a single-quarter performance  
            rather than one that summarizes performance across multiple quarters.
                
            **Instructions**:
            - Read the document chunk carefully.
            - Identify all product or business segments explicitly listed in the document.
            - Ensure that all distinct segments explicitly listed in the document are included in the segments, regardless of their position in the hierarchy.                 
            - Use internal reasoning to verify whether each field in answer is accurate, supported, and consistent with the text.
            - Validate the period is the most recent quarter available in the input table. 
            - Use the document chunk to refine and correct the answer so it is fully accurate, consistent, and supported by the document.
            - Output the corrected answer only in JSON format.
                          
            Output Example:
            {{
                    "period": "1994-06-30",
                    "currency": "USD",
                    "scale": "thousands",
                    "product_segments":
                        {{
                        "Mobile" : 1020,
                        "Service": 112
                        }},
                    "total_revenue" : 1132,
                    "reasoning": "<Describe how you validate the answer.>"
            }}
        
            user input:
            """).strip()
        else:
            return dedent(
            """
            You are given a document chunk and an answer derived from it.
            Your job is to internally verify the correctness of each field based on the document and output only the validated structured result.
            Always check missing products or business segments if any!!!  
            
            ## Annual Disambiguation Rule
            In 10-K filings, tables typically present **full fiscal year** results, sometimes alongside comparative prior years.

            - Identify the data that reflects a **complete fiscal year**.  
            Annual results are typically labeled with phrases such as  
            “for the year ended,” “for the twelve months ended,” or “for the fiscal year ended.”

            - Distinguish these from shorter or cumulative spans,  
            such as “three months ended” or “nine months ended,” which represent partial periods and should be ignored.

            - When multiple annual periods appear (for example, current and prior years),  
            select the one associated with the **latest fiscal year** in the table.

            - Always prefer the column that represents **a full-year performance**  
            rather than interim or partial periods.

            **Instructions**:
            - Read the document chunk carefully.
            - Identify all product or business segments explicitly listed in the document.
            - Ensure that all distinct segments explicitly listed in the document are included in the `product_segments` section, regardless of hierarchy or indentation.
            - Use internal reasoning to verify whether each field in `answer` is accurate, supported, and consistent with the text.
            - Validate that the period corresponds to the most recent **fiscal year** available in the input table.
            - Use the document chunk to refine and correct the answer so it is fully accurate, consistent, and supported by the document.
            - Output the corrected answer only in JSON format.
            
            ##Output Example:
            {{
                    "period": "1994-06-30",
                    "currency": "USD",
                    "scale": "thousands",
                    "product_segments":
                        {{
                        "Mobile" : 1020,
                        "Service": 112
                        }},
                    "total_revenue" : 1132,
                    "reasoning": "<Describe how you validate the answer.>"
            }}
        
            user input:
            """).strip()
    
    @staticmethod
    def checksum():
        return dedent("""
        You are a financial consistency checker.
        Your task is to determine whether a JSON object describing product segment revenues is internally consistent with the reported total revenue.

        ---

        ## Input format
        {
        "period": "...",
        "currency": "...",
        "scale": "...",
        "product_segments": { "<segment_name>": <value>, ... },
        "total_revenue": <number>
        }

        ---

        ## Instructions (follow strictly)

        1. **Read all numeric values.**
        Treat every key in "product_segments" as a segment name and its number as a revenue value.

        2. **Detect aggregates.**
        For each segment `X`:
        - Check if two or more *other* segment values sum approximately (±1%) to `X`.
        - If so, mark `X` as `"aggregate"`.
        - Otherwise, mark it as `"leaf"`.

        Example pattern (don’t memorize names):
        - If “A revenue” + “B revenue” ≈ “C revenue”, then C is aggregate.

        3. **Validate total_revenue.**
        - Sum all `"leaf"` segments.
        - If their sum ≈ total_revenue (within ±1%), data is valid.
        - Otherwise, check if any subset of segments approximately equals total_revenue.

        4. **Output format**
        Respond **only** with a JSON object:

        ```json
        {
        "is_valid": true or false,
        "aggregate_segments": ["..."],
        "leaf_segments": ["..."],
        "sum_leaf": <number>,
        "total_revenue": <number>,
        "difference": <number>,
        "reasoning": "<short numeric reasoning, e.g., which sums matched>"
        }

        **Rules to remember**

        * Use approximate numeric comparison (±1% tolerance).
        * Prefer smallest consistent set of leaf segments.
        * Don’t include text outside JSON.

        Input:

        """).strip()

    # "Ensure that all segments listed in the document are included in the answer."

# D:\Side_projects\llm_cache_test\result\GOOGL\predictions
# D:\Side_projects\llm_cache_test\preprocessed\GOOGL