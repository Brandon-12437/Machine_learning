import pickle


with open("pipeline_v1.bin", "rb") as f_in:
    model = pickle.load(f_in)


record = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}


prob = model.predict_proba([record])[0, 1]

print(f'Probability of conversion: {prob}')
