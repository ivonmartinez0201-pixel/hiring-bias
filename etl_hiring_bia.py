import pandas as pan
import numpy as num
import pymysql
from sqlalchemy import create_engine, text

csvdat = "AI_Hiring_Bias_Dataset.csv"
bdcon = {
    'host': 'localhost',
    'user': 'nico',
    'password': '123',
    'database': 'hiringbia'
}

print("Extrayendo datos desde:", csvdat)
dtf = pan.read_csv(csvdat)
print("Registros extraídos:", len(dtf))
print("Columnas:", dtf.columns.tolist())

print("**ETL** Iniciando transformación...")

dtf['candidate_id'] = dtf['candidate_id'].str.strip()
dtf['gender'] = dtf['gender'].str.strip().str.title()
dtf['education_level'] = dtf['education_level'].str.strip()
dtf['university_tier'] = dtf['university_tier'].str.strip().str.upper()

gender_map = {
    'Male': 'Male',
    'Female': 'Female',
    'Non-Binary': 'Non-Binary',
    'Nonbinary': 'Non-Binary',
    'Non Binary': 'Non-Binary'
}
dtf['gender'] = dtf['gender'].map(gender_map).fillna('Unknown')

dtf['university_tier'] = dtf['university_tier'].str.replace(' ', '_')

edu_map = {
    "Bachelor's": "Bachelor",
    "Bachelor": "Bachelor",
    "Bachelors": "Bachelor",
    "Master's": "Master",
    "Master": "Master",
    "Masters": "Master",
    "PhD": "PhD",
    "Ph.D": "PhD",
    "Doctorate": "PhD"
}
dtf['education_level'] = dtf['education_level'].map(edu_map).fillna('Other')

float_cols = [
    'years_experience', 'employment_gap_months',
    'technical_skill_score', 'communication_score',
    'aptitude_test_score', 'coding_test_score',
    'github_activity_score', 'ai_resume_score', 'ai_bias_score'
]
for col in float_cols:
    if col in dtf.columns:
        dtf[col] = pan.to_numeric(dtf[col], errors='coerce')

int_cols = ['age', 'project_count', 'certifications_count', 'expected_salary']
for col in int_cols:
    if col in dtf.columns:
        dtf[col] = pan.to_numeric(dtf[col], errors='coerce')

dtf['hired'] = dtf['hired'].apply(
    lambda x: 1 if x in [1, True, '1', 'True', 'true'] else 0
)

dtf.loc[dtf['age'] < 18, 'age'] = None
dtf.loc[dtf['age'] > 80, 'age'] = None

score_cols = ['technical_skill_score', 'communication_score', 
              'aptitude_test_score', 'coding_test_score',
              'github_activity_score', 'ai_resume_score', 'ai_bias_score']
for col in score_cols:
    if col in dtf.columns:
        dtf.loc[dtf[col] < 0, col] = None
        dtf.loc[dtf[col] > 100, col] = None

dtf.loc[dtf['years_experience'] < 0, 'years_experience'] = None

dtf = dtf.replace({num.nan: None, num.inf: None, -num.inf: None})

print("**ETL** Registros después de limpieza:", len(dtf))
print("**ETL** Registros con nulos:", dtf.isnull().any(axis=1).sum())

print("**ETL** Conectando a MySQL...")

try:
    engine = create_engine(
        f"mysql+pymysql://{bdcon['user']}:{bdcon['password']}@"
        f"{bdcon['host']}/{bdcon['database']}"
    )
    
    print("**ETL** Cargando datos a tabla 'candidates'...")
    dtf.to_sql(
        name='candidates',
        con=engine,
        if_exists='replace',
        index=False,
        chunksize=1000
    )

    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM candidates")) 
        count = result.fetchone()[0]
        print(f"**ETL** {count} registros cargados en MySQL")

    print("**ETL** Terminado")
    
except Exception as e:
    print(f"**ETL** Error en carga: {e}")