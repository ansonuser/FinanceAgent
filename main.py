from models.scraper import Extractor, main
import asyncio

ready = {'AMZN', 'MSFT', "NVDA", "AAPL", "GOOGL", "TSLA"}
MAX_CONCURRENT = 3
extractor = Extractor()
extractor.set_ollama(True)
try:
    asyncio.run(main(ready, extractor))
except KeyboardInterrupt:
    print("\n🛑 Interrupted by user.")