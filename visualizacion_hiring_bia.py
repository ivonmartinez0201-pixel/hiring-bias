import streamlit as strel
import pandas as pan
import numpy as num
import plotly.express as px
import plotly.graph_objects as go

strel.set_page_config(
    page_title="Análisis Sesgo IA",
    layout="wide"
)

dtf = pan.read_csv('datos_procesados.csv')
kp = pan.read_csv('indicadores.csv')
gen_summary = pan.read_csv('resumen_genero.csv')

strel.title("Análisis de Sesgo en Contratación con Inteligencia Artificial")
strel.markdown("""
**Contexto del Análisis**
Este proyecto examina el comportamiento de un sistema de inteligencia artificial diseñado para apoyar decisiones de contratación en el sector tecnológico. 
El estudio se centra en identificar posibles sesgos en las puntuaciones asignadas por el algoritmo a distintos grupos de candidatos.
""")

strel.sidebar.header("Filtros")

genders = ['Todos'] + sorted(dtf['gender'].unique().tolist())
sel_gender = strel.sidebar.selectbox("Género", genders)

tiers = ['Todos'] + sorted(dtf['university_tier'].unique().tolist())
sel_tier = strel.sidebar.selectbox("Tier Universitario", tiers)

min_sal = int(dtf['expected_salary'].min())
max_sal = int(dtf['expected_salary'].max())
sal_range = strel.sidebar.slider(
    "Rango de Salario Esperado",
    min_sal, max_sal,
    (min_sal, max_sal)
)

df_filt = dtf.copy()
if sel_gender != 'Todos':
    df_filt = df_filt[df_filt['gender'] == sel_gender]
if sel_tier != 'Todos':
    df_filt = df_filt[df_filt['university_tier'] == sel_tier]
df_filt = df_filt[
    (df_filt['expected_salary'] >= sal_range[0]) &
    (df_filt['expected_salary'] <= sal_range[1])
]

strel.header("Indicadores Clave")

col1, col2, col3, col4 = strel.columns(4)

with col1:
    hire_rate = (df_filt['hired'].sum() / len(df_filt)) * 100
    strel.metric("Tasa Contratación", f"{hire_rate:.1f}%")

with col2:
    total = len(df_filt)
    strel.metric("Total Candidatos", f"{total:,}")

with col3:
    gen_rates = df_filt.groupby('gender')['hired'].mean()
    if 'Male' in gen_rates.index and 'Female' in gen_rates.index:
        di = gen_rates.get('Female', 0) / gen_rates.get('Male', 1)
        strel.metric("DI (Female/Male)", f"{di:.3f}")
    else:
        strel.metric("DI (Female/Male)", "N/A")

with col4:
    avg_bias = df_filt['ai_bias_score'].mean()
    strel.metric("AI Bias Score Promedio", f"{avg_bias:.1f}")
    
strel.header("Análisis de Sesgo por Grupo")

col1, col2 = strel.columns(2)

with col1:
    strel.subheader("Tasa de Contratación por Género")
    gen_hire = df_filt.groupby('gender')['hired'].mean() * 100
    gen_hire = gen_hire.sort_values(ascending=False)
    
    fig1 = px.bar(
        x=gen_hire.index,
        y=gen_hire.values,
        color=gen_hire.index,
        text=gen_hire.values.round(1),
        title="Tasa de Contratación por Género",
        labels={'x': 'Género', 'y': 'Tasa de Contratación (%)'}
    )
    fig1.update_traces(textposition='outside')
    strel.plotly_chart(fig1, use_container_width=True)

with col2:
    strel.subheader("Tasa de Contratación por Tier")
    tier_hire = df_filt.groupby('university_tier')['hired'].mean() * 100
    tier_hire = tier_hire.sort_values(ascending=False)
    
    fig2 = px.bar(
        x=tier_hire.index,
        y=tier_hire.values,
        color=tier_hire.index,
        text=tier_hire.values.round(1),
        title="Tasa de Contratación por Tier Universitario",
        labels={'x': 'Tier', 'y': 'Tasa de Contratación (%)'}
    )
    fig2.update_traces(textposition='outside')
    strel.plotly_chart(fig2, use_container_width=True)

strel.header("Análisis de AI Bias Score")

col1, col2 = strel.columns(2)

with col1:
    strel.subheader("Distribución de AI Bias Score por Género")
    fig3 = px.box(
        df_filt,
        x='gender',
        y='ai_bias_score',
        color='gender',
        title="Distribución de AI Bias Score por Género",
        labels={'gender': 'Género', 'ai_bias_score': 'AI Bias Score'}
    )
    strel.plotly_chart(fig3, use_container_width=True)

with col2:
    strel.subheader("AI Bias Score vs Skill Técnico")
    fig4 = px.scatter(
        df_filt,
        x='technical_skill_score',
        y='ai_bias_score',
        color='hired',
        title="Relación: Skill Técnico vs AI Bias Score",
        labels={
            'technical_skill_score': 'Skill Técnico',
            'ai_bias_score': 'AI Bias Score',
            'hired': 'Contratado'
        },
        color_discrete_map={0: 'red', 1: 'green'}
    )
    strel.plotly_chart(fig4, use_container_width=True)

strel.header("Matriz de Correlación")

corr_vars = ['ai_bias_score', 'technical_skill_score', 
             'communication_score', 'years_experience', 'hired']
corr_mat = df_filt[corr_vars].corr()

fig5 = go.Figure(data=go.Heatmap(
    z=corr_mat.values,
    x=corr_mat.columns,
    y=corr_mat.index,
    colorscale='RdBu_r',
    zmin=-1,
    zmax=1,
    text=corr_mat.values.round(3),
    texttemplate='%{text}',
    textfont={"size": 12}
))
fig5.update_layout(
    title="Matriz de Correlación",
    width=700,
    height=600
)
strel.plotly_chart(fig5, use_container_width=True)

strel.header("Resultados Estadísticos")

col1, col2 = strel.columns(2)

with col1:
    strel.subheader("Indicadores Clave")
    strel.dataframe(kp, use_container_width=True)

with col2:
    strel.subheader("Resumen por Género")
    strel.dataframe(gen_summary, use_container_width=True)

with strel.expander("Ver Datos Filtrados"):
    strel.dataframe(df_filt, use_container_width=True)
    strel.caption(f"Mostrando {len(df_filt)} registros de {len(dtf)} totales")
    
strel.markdown("---")
