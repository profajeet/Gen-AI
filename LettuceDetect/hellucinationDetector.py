# pip install lettucedetect -U
from lettucedetect.models.inference import HallucinationDetector

# For English:
detector = HallucinationDetector(
    method="transformer", 
    model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1",
)

# For other languages (e.g., German):
# detector = HallucinationDetector(
#     method="transformer", 
#     model_path="KRLabsOrg/lettucedect-210m-eurobert-de-v1",
#     lang="de",
#     trust_remote_code=True
# )

contexts = ["France is a country in Europe. The capital of France is Paris. The population of France is 67 million.",]
question = "What is the capital of France? What is the population of France?"
answer = "The capital of France is Paris. The population of France is 69 million."

# Get span-level predictions indicating which parts of the answer are considered hallucinated.
predictions = detector.predict(context=contexts, question=question, answer=answer, output_format="spans")
print("Predictions:", predictions)

# Predictions: [{'start': 31, 'end': 71, 'confidence': 0.9944414496421814, 'text': ' The population of France is 69 million.'}]