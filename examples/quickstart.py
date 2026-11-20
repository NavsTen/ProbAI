"""Run with:  python examples/quickstart.py"""

from probai import MissingParametersError, ProbAI

ai = ProbAI()

questions = [
    "What is the probability of rolling a 7 with two dice?",
    "Probability of at least one head in 3 coin tosses",
    "A call center receives 12 calls per hour on average. Probability of exactly 15 calls in an hour?",
    "Probability of bit error on a BPSK radio link at 8 dB",
    "What is the probability of correctly receiving a signal, given that the signal-to-noise ratio is 0.8?",
    "Probability that a borrower defaults within 3 years with an annual default rate of 2%",
    "A test has 99% sensitivity and 95% specificity, prevalence is 1%. I tested positive: chance I have it?",
    "Chance that a drive with a mean lifetime of 10 years fails within 5 years",
    "We saw 37 conversions out of 400 visitors; chance the next visitor converts?",
    "Probability of bit error on a radio link",  # no number given on purpose
    "Какова вероятность правильно принять сигнал, если отношение сигнал/шум равно 0,8?",
    "Колл-центр получает в среднем 12 звонков в час. Какова вероятность ровно 15 звонков за час?",
]

for text in questions:
    print(f"\nQ: {text}")
    try:
        est = ai.estimate(text)
        print(f"   P = {est.probability:.6g}  [{est.engine}, routing confidence {est.routing_confidence:.0%}]")
        if est.extracted:
            print(f"   understood: {est.extracted}")
        for note in est.notes:
            print(f"   note: {note}")
    except MissingParametersError as err:
        print(f"   needs more data: {err}")
