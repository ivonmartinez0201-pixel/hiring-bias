import pandas as pan
import numpy as num
from sqlalchemy import create_engine, text
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
from scipy import stats

bdcon = {
    'host': 'localhost',
    'user': 'nico',
    'password': '123',
    'database': 'hiringbia'
}

eng = create_engine(
    f"mysql+pymysql://{bdcon['user']}:{bdcon['password']}@"
    f"{bdcon['host']}/{bdcon['database']}"
)

with eng.connect() as conn:
    dtf = pan.read_sql(text("SELECT * FROM candidates"), conn)

print(f"Total registros: {len(dtf)}")
print(f"Columnas: {dtf.columns.tolist()}")

print("Indicadores...")

hired_rate = (dtf['hired'].sum() / len(dtf)) * 100
print(f"Tasa de contratación general: {hired_rate:.2f}%")

gen_hire = dtf.groupby('gender')['hired'].agg(['count', 'mean'])
gen_hire['hire_pct'] = gen_hire['mean'] * 100
print("Tasa de contratación por género:")
print(gen_hire[['count', 'hire_pct']])

male_rate = gen_hire.loc['Male', 'hire_pct'] / 100
print("Disparidad de Impacto (DI):")
for g in gen_hire.index:
    if g != 'Male':
        di = (gen_hire.loc[g, 'hire_pct'] / 100) / male_rate
        status = "SESGO" if di < 0.8 else "OK"
        print(f"  {g}: {di:.3f} → {status}")

tier_hire = dtf.groupby('university_tier')['hired'].mean() * 100
print("Tasa de contratación por Tier Universitario:")
print(tier_hire.sort_values(ascending=False))

bias_gen = dtf.groupby('gender')['ai_bias_score'].mean()
print("AI Bias Score promedio por género:")
print(bias_gen)

corr_vars = ['ai_bias_score', 'technical_skill_score', 
             'communication_score', 'years_experience', 'hired']
corr_mat = dtf[corr_vars].corr()
print("Matriz de correlación:")
print(corr_mat.round(3))

print("Prueba Chi-cuadrado (Género vs Contratación)")
tbl_gen = pan.crosstab(dtf['gender'], dtf['hired'])
chi2, p_val, dof, exp = chi2_contingency(tbl_gen)
print(f"  Chi2: {chi2:.4f}")
print(f"  p-value: {p_val:.6f}")
if p_val < 0.05:
    print("Existe relación significativa entre género y contratación")
else:
    print("No existe evidencia de relación")

print("Prueba t (AI Bias Score: Contratados vs No Contratados)")
hired_bias = dtf[dtf['hired']==1]['ai_bias_score']
not_hired_bias = dtf[dtf['hired']==0]['ai_bias_score']
t_stat, p_t = ttest_ind(hired_bias, not_hired_bias)
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_t:.6f}")
if p_t < 0.05:
    print("Existe diferencia significativa en AI Bias Score")
else:
    print("No existe diferencia significativa")

print("ANOVA (AI Bias Score por Tier Universitario)")
tier_groups = [dtf[dtf['university_tier']==t]['ai_bias_score'].dropna() 
               for t in dtf['university_tier'].unique()]
f_stat, p_anova = f_oneway(*tier_groups)
print(f"  F-statistic: {f_stat:.4f}")
print(f"  p-value: {p_anova:.6f}")
if p_anova < 0.05:
    print("Existe diferencia significativa entre tiers")
else:
    print("No existe diferencia significativa")

print("Correlación de Pearson (AI Bias Score vs Contratación)")
corr_pearson, p_corr = stats.pearsonr(dtf['ai_bias_score'], dtf['hired'])
print(f"  Correlación: {corr_pearson:.4f}")
print(f"  p-value: {p_corr:.6f}")
if p_corr < 0.05:
    print("Existe correlación significativa")
else:
    print("No existe correlación significativa")

print("Resumen por género:")
gen_summary = dtf.groupby('gender').agg({
    'age': ['mean', 'std'],
    'years_experience': ['mean', 'std'],
    'technical_skill_score': ['mean', 'std'],
    'ai_bias_score': ['mean', 'std'],
    'hired': ['mean']
}).round(2)
print(gen_summary)

print("Resumen por Tier Universitario:")
tier_summary = dtf.groupby('university_tier').agg({
    'technical_skill_score': ['mean', 'std'],
    'ai_bias_score': ['mean', 'std'],
    'hired': ['mean']
}).round(2)
print(tier_summary)

dtf.to_csv('datos_procesados.csv', index=False)
print("Datos procesados guardados en: datos_procesados.csv")

kpi_df = pan.DataFrame({
    'indicador': ['Tasa_Contratacion_General', 'Chi2_pvalue', 't_test_pvalue', 'ANOVA_pvalue'],
    'valor': [hired_rate, p_val, p_t, p_anova]
})
kpi_df.to_csv('indicadores.csv', index=False)
print("Indicadores guardados en: indicadores.csv")

gen_hire.to_csv('resumen_genero.csv')
print("Resumen por género guardado en: resumen_genero.csv")

print("Análisis Terminado")