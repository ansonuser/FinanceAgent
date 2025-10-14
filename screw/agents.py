from typing import Union, Dict, Optional, List
from utils.utils import call_api, extract_json
from schema.data import RevenueData
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential_jitter,
)
from tenacity import stop_after_attempt
from json import JSONDecodeError

retryable_exceptions = (
    JSONDecodeError,
    TypeError
)
from textwrap import dedent

def on_retry_error(retry_state):
    # This is called when retries are exhausted
    print("All retries failed — continuing gracefully.")
    return {}  # Fallback result


class Extrator:
    """
    A wrapper to extract data from financial
    """
    def __init__(self, sys_prompt: str = None, base_url: str = None, model_name: str = "qwen2.5-coder:14b"):
        if sys_prompt:
            self._sys_prompt = sys_prompt
        else:
            self.set_sys()
        if base_url:
            self._base_url = base_url.rstrip('/') 
        else:
            self._base_url = "http://localhost:11434/api/chat"

        self._model_name = model_name
    
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
                    {"role" : "user", "content" : self._sys_prompt + '\n' + prompts[0]["content"]},
                ] 
                messages += prompts[1:]
        else:
            messages = prompts

        # messages = [{"role": "system", "content" : "You are a helpful assistant"}] + messages 
        # print("messages=", messages)
        res = call_api(messages, self._base_url, self._model_name)
        # print(res)
        if schema:
            json_res = extract_json(res)
            if "currency" not in json_res:
                json_res["currency"] = "USD"
            return schema(**json_res)
        return res
    
    def set_sys(self):
        # 2. The leftmost value from the table is likely the answer compare to other values in same row.

        prompt = """
        You are an expert financial report parser. Your task is to extract revenue by product segment from the given text.
        
        ## Rule for Disambiguation in 10-Q Tables
        * Header context: Use the preceding phrase (e.g. “Three Months Ended ,”) to bind each year.
        * Canonical form: Combine header + year (e.g."Three Months Ended June 30 2025" and "Three Months Ended June 30 2024".)        
        ### Normalization logic:
        * Only focus on quarter period and ignore any period longer than quarter. 
        * If the table contains only historical years, treat the latest available year as the Current Quarter for extraction purposes.   

        Follow these rules carefully:

        1. Target Information
        * Extract revenue by product segment for the most recent quarter available in the input table (the latest year shown).
        * Extract all revenue line items by business segment or product category, e.g. “Payments and other fees”, “Cloud”, “Hardware”, etc.
        * Exclude all percentage signs, labels like “%”, and any percentage-based data.
        * Include their values, currencies, and periods.
        * Include total revenue, if available.
        * Reject ratio-based or percentage-mostly tables.
        * Ignore unrelated metrics (e.g., costs, compensation, R&D, etc.).

        2. Normalization
        * Normalize numbers by removing “$”, commas, and parentheses.
        * Interpret “(5)” as –5.
        * Convert units (e.g. "in millions" to "millions") appropriately if mentioned.
        * Maintain consistent period labels.(See Rule for Disambiguation in 10-Q Tables)

        3. Output Format
        Return only a VALID JSON object, the commentary must go in the reasoning field.
        Leave the value empty if not value found.
        Follow exactly this schema:

        {
            "period": "2025 Q1",
            "currency": "USD",
            "unit": "thousands",
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
        * Pay attention on `unit`, especially not include percentage-based data.

        Now analyze the following input.
        """
        self._sys_prompt = dedent(prompt)
    @staticmethod
    def send_validation(company_name : str)->str:
        return dedent(
        f"""
        Identify the correct quarterly revenue segment by product for {company_name}
        Think in the following process:
        1. Determine how {company_name} earned revenue using primary sources
        2. List {company_name}’s key products
        3. It eithers all keys in segments are valid or invalid.
        Internally analyze (without showing reasoning) and verify the correct answer and output it as JSON format at the end.
        """)
    
    @staticmethod
    def send_refinement():
        #   - Confirm the values are from the column labeled as Three Months Ended September 30, 2024
        return dedent("""
        You are given a document chunk and an answer derived from it.
        Your job is to internally verify the correctness of each field based on the document and output only the validated structured result.
        Always check missing products or business segemnts if any!!!  
        
        ## Rule for Disambiguation in 10-Q Tables
        * Header context: Use the preceding phrase (e.g. “Three Months Ended June 30,”) to bind each year.
        * Canonical form: Combine header + year (e.g."Three Months Ended September 30 2025" and "Three Months Ended September 30 2024".)
        ### Normalization logic:
        * Only focus on quarter period and ignore any period longer than quarter. 
        * If the table contains only historical years, treat the latest available year as the Current Quarter for extraction purposes.   
  
        **Instructions**:
        - Read the document chunk carefully.
        - Identify all product or business segments explicitly listed in the document.
        - Ensure that all distinct segments explicitly listed in the document are included in the segments, regardless of their position in the hierarchy.                 
        - Use internal reasoning to verify whether each field in answer is accurate, supported, and consistent with the text.
        - Validate the period is the most recent quarter available in the input table. 
        - Use the document chunk to refine and correct the answer so it is fully accurate, consistent, and supported by the document.
        - Output the corrected answer only in JSON format.
    
        user input:
        """)
    
    # "Ensure that all segments listed in the document are included in the answer."

