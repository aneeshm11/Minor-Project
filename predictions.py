import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

df = pd.read_csv(r"C:\Users\Aneesh\Downloads\CODE\project\Crystal_structure.csv")
df = df.drop(['In literature', 'Compound'], axis=1)
col=['Compound', 'A', 'B', 'In literature', 'v(A)', 'v(B)', 'r(AXII)(Å)', 'r(AVI)(Å)', 'r(BVI)(Å)', 'EN(A)', 'EN(B)', 'l(A-O)(Å)', 'l(B-O)(Å)', 'ΔENR', 'tG', 'τ', 'μ', 'Lowest distortion']

df_cleaned = df.fillna(0)
df_cleaned = df_cleaned.replace('-', 0)

df_encoded = pd.get_dummies(df_cleaned, columns=['A', 'B', 'Lowest distortion'])
dfn = df_encoded.drop("Lowest distortion_0", axis=1)


xtrain = dfn.iloc[:, :159]


#  refer the input.txt file to modify the inputs for prediction


data = {
    'A': 'Ac',
    'B': 'Ac',
    'v(A)': 0,
    'v(B)': 0,
    'r(AXII)(Å)': 1.12,
    'r(AVI)(Å)': 1.12,
    'r(BVI)(Å)': 1.12,
    'EN(A)': 1.1,
    'EN(B)': 1.1,
    'l(A-O)(Å)': 0.0,
    'l(B-O)(Å)': 0.0,
    'ΔENR': -3.248,
    'tG': 0.707107,
    'τ': '-',
    'μ': 0.8
}

input_df = pd.DataFrame(data, index=[0])
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=xtrain.columns, fill_value=0)
input_values = input_encoded.values
x=np.array(input_values, dtype=np.float32).reshape((1,-1))


model = load_model("data.h5")
out= (model.predict(x))[0]



def func(user_in):
    class_names = ['Lowest distortion_cubic', 'Lowest distortion_orthorhombic', 'Lowest distortion_rhombohedral', 'Lowest distortion_tetragonal']

    for prob, class_name in zip(user_in, class_names):
        print(f"Probability of compound to have {class_name} crystal structure is : {prob * 100 } % ")

print(func(out))
