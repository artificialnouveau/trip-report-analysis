import pandas as pd
import random
import faker

# Set the seed for reproducibility
random.seed(42)
fake = faker.Faker()

# Define lists for data
genders = ['Male', 'Female', 'Non-binary', 'Prefer not to say']
drugs = ['LSD']
ages = list(range(18, 65))
dosages = list(range(100, 500, 50))

# Generate data
data = {
    'Gender': [random.choice(genders) for _ in range(50)],
    'Age': [random.choice(ages) for _ in range(50)],
    'Description': [fake.text(max_nb_chars=100) for _ in range(50)],
    'Drug': [random.choice(drugs) for _ in range(50)],
    'Date': [fake.date_this_century() for _ in range(50)],
    'Dosage_mcg': [random.choice(dosages) for _ in range(50)],
}

# Create DataFrame
df = pd.DataFrame(data)

print(df)
