import pandas as pd
import glob
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

fuel_map = {
    'BATTERY OPERATED': 'ELECTRIC',
    'BATTERY': 'ELECTRIC',
    'DIESEL ': 'DIESEL',
    'PETROL/LPG': 'PETROL LPG',
    'LPG PETROL': 'PETROL LPG',
    'LPG  PETROL': 'PETROL LPG',
    'PETROL CNG': 'CNG PETROL',
    'PL': 'PETROL LPG',
    '-1': 'UNKNOWN',
}

col_map = {
    'registrationNo': 'Regn_No',
    'regvalidfrom':   'Regn_VFDt',
    'regvalidto':     'Regn_VTDt',
    'makerName':      'Maker_Name',
    'modelDesc':      'Model_Desc',
    'bodyType':       'BodyTyp',
    'cc':             'CC',
    'cylinder':       'Cylinder',
    'fuel':           'Fuel',
    'hp':             'HP',
    'seatCapacity':   'Seat_Capacity',
    'OfficeCd':       'OfficeCd',
}

all_files = glob.glob("data/*.csv")
print(f"Found {len(all_files)} files\n")

dfs = []
for f in all_files:
    for enc in ["utf-8", "cp1252", "latin1"]:
        try:
            temp = pd.read_csv(f, encoding=enc)
            temp = temp.rename(columns=col_map)
            dfs.append(temp)
            print(f"Loaded: {f} - {len(temp)} rows")
            break
        except Exception:
            continue

if not dfs:
    raise RuntimeError("No CSV files could be loaded from data/*.csv")

df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal rows: {len(df)}")

df['Fuel_Clean'] = df['Fuel'].astype(str).str.strip().str.upper()
df['Fuel_Clean'] = df['Fuel_Clean'].replace(
    {k.strip().upper(): v for k, v in fuel_map.items()}
)

print("\nFuel_Clean counts:")
print(df['Fuel_Clean'].value_counts())

electric_count = (df['Fuel_Clean'] == 'ELECTRIC').sum()
print(f"\nTotal ELECTRIC vehicles across all files: {electric_count}")

print("\nEV by year:")
df['Regn_VFDt'] = pd.to_datetime(df['Regn_VFDt'], errors='coerce')
df['Year'] = df['Regn_VFDt'].dt.year
df_clean = df[(df['Year'] >= 2018) & (df['Year'] <= 2026)]
ev_by_year = df_clean[df_clean['Fuel_Clean'] == 'ELECTRIC'].groupby('Year').size()
total_by_year = df_clean.groupby('Year').size()
print((ev_by_year / total_by_year * 100).round(3))

df['Is_Electric'] = (df['Fuel_Clean'] == 'ELECTRIC').astype(int)

features = ['Maker_Name', 'BodyTyp', 'CC', 'OfficeCd']
df_model = df[features + ['Is_Electric']].dropna()
df_model['CC'] = pd.to_numeric(df_model['CC'], errors='coerce')
df_model = df_model.dropna()

le_maker = LabelEncoder()
le_body = LabelEncoder()
le_office = LabelEncoder()

df_model['Maker_enc'] = le_maker.fit_transform(df_model['Maker_Name'])
df_model['Body_enc'] = le_body.fit_transform(df_model['BodyTyp'])
df_model['Office_enc'] = le_office.fit_transform(df_model['OfficeCd'])

X = df_model[['Maker_enc', 'Body_enc', 'CC', 'Office_enc']]
y = df_model['Is_Electric']

print(f"Electric: {y.sum()}, Non-electric: {(y==0).sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nModel results:")
print(classification_report(y_test, y_pred))

feat_importance = pd.Series(
    model.feature_importances_,
    index=['Maker', 'BodyType', 'CC', 'Office']
).sort_values(ascending=False)
print("\nFeature importance:")
print(feat_importance)

# Experiment: remove Maker to test dependency
X_no_maker = df_model[['Body_enc', 'CC', 'Office_enc']]

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_no_maker, y, test_size=0.2, random_state=42, stratify=y
)

model2 = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model2.fit(X_train2, y_train2)
y_pred2 = model2.predict(X_test2)

print("\nModel WITHOUT Maker_Name:")
print(classification_report(y_test2, y_pred2))

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/ev_classifier.pkl")
joblib.dump(le_maker, "model/le_maker.pkl")
joblib.dump(le_body, "model/le_body.pkl")
joblib.dump(le_office, "model/le_office.pkl")

print("Model saved successfully.")
print(f"Known makers: {len(le_maker.classes_)}")
print(f"Known body types: {list(le_body.classes_)}")
print(f"Known offices: {len(le_office.classes_)}")

# Experiment: remove Maker to test dependency
print("\n--- EXPERIMENT: Model WITHOUT Maker_Name ---")
X_no_maker = df_model[['Body_enc', 'CC', 'Office_enc']]

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_no_maker, y, test_size=0.2, random_state=42, stratify=y
)

model2 = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model2.fit(X_train2, y_train2)
y_pred2 = model2.predict(X_test2)
print(classification_report(y_test2, y_pred2))