import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import io
import base64

# Gestion des imports optionnels
try:
    from pyreadstat import read_sav, write_sav
    PYREADSTAT_AVAILABLE = True
except ImportError:
    PYREADSTAT_AVAILABLE = False

try:
    from utils.analysis_functions import *
    from utils.data_generator import DataGenerator
except ImportError:
    # Fallback si les modules ne sont pas disponibles
    PYREADSTAT_AVAILABLE = False
    
    class DataGenerator:
        def generate_complex_dataset(self, **kwargs):
            st.error("Générateur de données non disponible")
            return pd.DataFrame()

try:
    from templates.download_pages import show_download_section, show_data_preview, show_data_quality_report
except ImportError:
    def show_download_section(generated_data):
        st.warning("Module de téléchargement non disponible")
    
    def show_data_preview(generated_data):
        if generated_data is not None:
            st.dataframe(generated_data.head())
    
    def show_data_quality_report(generated_data):
        pass

from data_downloader import DataDownloader

# Configuration de la page
st.set_page_config(
    page_title="LogicApp Analytics Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé avec design professionnel
def load_css():
    st.markdown("""
    <style>
        /* Styles généraux */
        .main {
            background-color: #f8f9fa;
        }
        
        .main-header {
            font-size: 2.8rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
            font-weight: 700;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .sub-header {
            font-size: 1.2rem;
            color: #6c757d;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 300;
        }
        
        .section-header {
            font-size: 1.6rem;
            color: #495057;
            margin-top: 2rem;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #667eea;
            font-weight: 600;
        }
        
        .card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
            border-left: 4px solid #667eea;
        }
        
        .feature-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin: 0.5rem;
        }
        
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
            color: white;
        }
        
        .stButton button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        }
        
        .download-btn {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
        }
        
        .success-message {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        .warning-message {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        /* Footer */
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 1rem;
            text-align: center;
            font-size: 0.9rem;
            z-index: 1000;
        }
        
        .footer-content {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo-text {
            font-weight: 700;
            font-size: 1.1rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .main-header {
                font-size: 2rem;
            }
            
            .footer-content {
                flex-direction: column;
                gap: 0.5rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

def add_footer():
    st.markdown("""
    <div class="footer">
        <div class="footer-content">
            <div>
                <span class="logo-text">LabMath Analytics Pro</span> - Plateforme d'Analyse Scientifique
            </div>
            <div>
                🔬 Powered by <strong>Lab_Math SCSM</strong> and <strong>CIE</strong>
            </div>
            <div>
                © Copyright 2025 - Tous droits réservés
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Ajouter de l'espace pour éviter que le contenu soit caché par le footer
    st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)

def add_logo():
    """Ajoute un logo personnalisé"""
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: white; margin-bottom: 0.5rem;">🔬</h2>
        <h3 style="color: white; margin: 0;">LabMath Analytics</h3>
        <p style="color: #bdc3c7; font-size: 0.8rem; margin: 0;">Pro Edition</p>
    </div>
    """, unsafe_allow_html=True)

def show_welcome():
    """Page d'accueil avec présentation"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h1 class="main-header">🔬 LabMath Analytics Pro</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Plateforme Professionnelle d\'Analyse de Données Scientifiques</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>🚀 Bienvenue sur votre plateforme d'analyse avancée</h3>
            <p>Cette application vous permet d'effectuer des analyses statistiques complexes, 
            générer des visualisations interactives et exporter vos résultats dans tous les formats standards.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 12px; color: white; text-align: center;">
            <h3>📊 Fonctionnalités</h3>
            <p>• Analyse multidimensionnelle</p>
            <p>• Tests statistiques avancés</p>
            <p>• Visualisations interactives</p>
            <p>• Export multi-formats</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Cartes de fonctionnalités
    st.markdown("### 🎯 Fonctionnalités Principales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>📥 Données</h4>
            <p>Génération et import multi-formats</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Analyse</h4>
            <p>Statistiques descriptives et inférentielles</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>🎨 Visualisation</h4>
            <p>Graphiques interactifs et personnalisables</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <h4>📈 Reporting</h4>
            <p>Export professionnel et automatisation</p>
        </div>
        """, unsafe_allow_html=True)

def read_data_file(uploaded_file, file_extension):
    """
    Lit un fichier uploadé selon son format
    """
    try:
        if file_extension in ['xlsx', 'xls']:
            return pd.read_excel(uploaded_file)
        elif file_extension == 'csv':
            return pd.read_csv(uploaded_file)
        elif file_extension == 'dta':
            return pd.read_stata(uploaded_file)
        elif file_extension == 'sav':
            if PYREADSTAT_AVAILABLE:
                df, meta = read_sav(uploaded_file)
                return df
            else:
                st.error("❌ Le format SPSS (.sav) n'est pas supporté sur cette plateforme")
                st.info("💡 Utilisez Excel, CSV ou STATA à la place")
                return None
        elif file_extension == 'txt':
            # Essayer différents séparateurs
            try:
                return pd.read_csv(uploaded_file, delimiter='\t')
            except:
                return pd.read_csv(uploaded_file, delimiter=' ')
        else:
            st.error(f"❌ Format {file_extension} non supporté")
            return None
    except Exception as e:
        st.error(f"❌ Erreur lors de la lecture du fichier: {str(e)}")
        return None

def telecharger_donnees():
    st.markdown('<h2 class="section-header">📥 Télécharger des Données d\'Exemple</h2>', unsafe_allow_html=True)
    
    st.info("""
    **📋 Instructions :**
    - Téléchargez un jeu de données d'exemple dans le format de votre choix
    - Utilisez ces données pour tester l'application
    - Les données contiennent des variables catégorielles et numériques réalistes
    """)
    
    # Configuration des données à générer
    st.markdown("### ⚙️ Configuration des Données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_observations = st.slider("Nombre d'observations", 100, 5000, 1000)
        n_variables = st.slider("Nombre de variables", 5, 30, 15)
    
    with col2:
        include_missing = st.checkbox("Inclure des valeurs manquantes", value=True)
        missing_percentage = st.slider("Pourcentage de valeurs manquantes", 0.0, 20.0, 5.0) if include_missing else 0.0
    
    # Types de variables
    st.markdown("### 🎯 Types de Variables")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_categorical = st.number_input("Variables catégorielles", 1, n_variables-2, 5)
    with col2:
        n_numerical = st.number_input("Variables numériques", 1, n_variables-2, 7)
    with col3:
        n_binary = st.number_input("Variables binaires", 1, n_variables-2, 3)
    
    # Génération des données
    if st.button("🔄 Générer les Données", type="primary"):
        with st.spinner("Génération des données en cours..."):
            try:
                generator = DataGenerator()
                df_generated = generator.generate_complex_dataset(
                    n_observations=n_observations,
                    n_categorical=n_categorical,
                    n_numerical=n_numerical,
                    n_binary=n_binary,
                    missing_percentage=missing_percentage
                )
                
                st.session_state.generated_data = df_generated
                st.success(f"✅ Données générées avec succès! Shape: {df_generated.shape}")
            except Exception as e:
                st.error(f"❌ Erreur lors de la génération: {str(e)}")
    
    # Affichage des données générées
    if st.session_state.generated_data is not None:
        show_data_preview(st.session_state.generated_data)
        show_data_quality_report(st.session_state.generated_data)
        show_download_section(st.session_state.generated_data)

def charger_donnees():
    st.markdown('<h2 class="section-header">📁 Chargement des Données</h2>', unsafe_allow_html=True)
    
    # Option: utiliser les données générées ou uploader
    data_source = st.radio(
        "Source des données:",
        ["📤 Uploader un fichier", "🎯 Utiliser les données générées"]
    )
    
    if data_source == "📤 Uploader un fichier":
        # Upload de fichier
        uploaded_file = st.file_uploader(
            "Choisissez votre fichier de données",
            type=['xlsx', 'xls', 'csv', 'dta', 'txt'],
            help="Formats supportés: Excel, CSV, STATA, TXT"
        )
        
        if uploaded_file is not None:
            try:
                # Lecture du fichier selon l'extension
                file_extension = uploaded_file.name.split('.')[-1].lower()
                df = read_data_file(uploaded_file, file_extension)
                if df is not None:
                    st.session_state.df = df
                    st.success(f"✅ Données chargées avec succès! Shape: {df.shape}")
                
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement: {str(e)}")
    
    else:  # Utiliser les données générées
        if st.session_state.generated_data is not None:
            st.session_state.df = st.session_state.generated_data
            st.success("✅ Données générées chargées avec succès!")
        else:
            st.warning("⚠️ Veuillez d'abord générer des données dans la section 'Télécharger données'")
    
    # Aperçu et sélection des variables si des données sont chargées
    if st.session_state.df is not None:
        df = st.session_state.df
        
        with st.expander("👀 Aperçu des données"):
            st.dataframe(df.head())
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Informations sur les colonnes:**")
                st.write(f"Nombre de variables: {len(df.columns)}")
                st.write(f"Nombre d'observations: {len(df)}")
            
            with col2:
                st.write("**Types de données:**")
                type_counts = df.dtypes.value_counts()
                for dtype, count in type_counts.items():
                    st.write(f"- {dtype}: {count}")
        
        # Sélection des variables
        st.markdown("### 🎯 Sélection des Variables")
        
        toutes_variables = df.columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            var_interet = st.selectbox(
                "Variable d'intérêt (Var_intérêt):",
                options=toutes_variables,
                help="Variable dépendante ou variable de regroupement"
            )
        
        with col2:
            var_independantes = st.multiselect(
                "Variables indépendantes (Var_ind1, ..., Var_indn):",
                options=[v for v in toutes_variables if v != var_interet],
                help="Sélectionnez une ou plusieurs variables indépendantes"
            )
        
        if st.button("💾 Valider la sélection", type="primary"):
            st.session_state.var_interet = var_interet
            st.session_state.var_independantes = var_independantes
            st.success("✅ Sélection des variables validée!")

def repartition_variables():
    st.markdown('<h2 class="section-header">📊 Répartition de Toutes les Variables</h2>', unsafe_allow_html=True)
    
    df = st.session_state.df
    var_interet = st.session_state.var_interet
    
    if var_interet is None:
        st.warning("Veuillez d'abord sélectionner une variable d'intérêt")
        return
    
    # Options d'affichage
    col1, col2, col3 = st.columns(3)
    with col1:
        show_percentages = st.checkbox("Afficher les pourcentages", value=True)
    with col2:
        show_totals = st.checkbox("Afficher les totaux", value=True)
    with col3:
        max_categories = st.number_input("Max catégories par variable", min_value=5, max_value=50, value=15)
    
    # Générer les tableaux de répartition pour toutes les variables
    progress_bar = st.progress(0)
    all_tables = []
    
    variables_to_analyze = [col for col in df.columns if col != var_interet]
    
    for i, variable in enumerate(variables_to_analyze):
        table = generate_frequency_table(df, variable, var_interet, max_categories)
        all_tables.append((variable, table))
        progress_bar.progress((i + 1) / len(variables_to_analyze))
    
    # Afficher les tableaux
    for variable_name, table_df in all_tables:
        with st.expander(f"📋 {variable_name}", expanded=False):
            st.dataframe(table_df)
            
            # Option de téléchargement pour chaque tableau
            csv = table_df.to_csv()
            st.download_button(
                label=f"📥 Télécharger {variable_name}",
                data=csv,
                file_name=f"repartition_{variable_name}.csv",
                mime="text/csv",
                key=f"download_{variable_name}"
            )

def generate_frequency_table(df, variable, group_variable, max_categories=15):
    """
    Génère un tableau de fréquences avec effectifs et pourcentages
    """
    # Gérer les variables avec trop de catégories
    if df[variable].nunique() > max_categories:
        # Regrouper les catégories peu fréquentes
        value_counts = df[variable].value_counts()
        top_categories = value_counts.head(max_categories - 1).index
        df_temp = df.copy()
        df_temp[variable] = df_temp[variable].apply(
            lambda x: x if x in top_categories else 'Autres'
        )
    else:
        df_temp = df
    
    # Créer le tableau croisé
    cross_tab = pd.crosstab(
        df_temp[variable], 
        df_temp[group_variable],
        margins=True,
        margins_name="Total"
    )
    
    # Ajouter les pourcentages
    percent_tab = cross_tab.div(cross_tab.iloc[-1]) * 100
    
    # Combiner effectifs et pourcentages
    result_tab = cross_tab.astype(str) + " (" + percent_tab.round(2).astype(str) + "%)"
    
    return result_tab

def tableaux_croises():
    st.markdown('<h2 class="section-header">🔍 Tableaux Croisés</h2>', unsafe_allow_html=True)
    
    df = st.session_state.df
    var_interet = st.session_state.var_interet
    var_independantes = st.session_state.var_independantes
    
    if not var_independantes:
        st.warning("Veuillez sélectionner des variables indépendantes")
        return
    
    # Sélection des variables à croiser
    selected_vars = st.multiselect(
        "Choisir les variables à croiser avec la variable d'intérêt:",
        options=var_independantes,
        default=var_independantes[:2] if len(var_independantes) >= 2 else var_independantes
    )
    
    if selected_vars:
        # Options d'affichage
        col1, col2 = st.columns(2)
        with col1:
            display_type = st.radio(
                "Type d'affichage:",
                ["Effectifs", "Pourcentages en ligne", "Pourcentages en colonne", "Pourcentages totaux"]
            )
        with col2:
            show_chi2 = st.checkbox("Afficher test du Chi²", value=True)
        
        # Générer les tableaux croisés
        for var in selected_vars:
            st.markdown(f"### 📊 Croisement: {var_interet} × {var}")
            
            # Tableau croisé
            cross_table = pd.crosstab(df[var], df[var_interet])
            
            if display_type == "Pourcentages en ligne":
                cross_table_display = cross_table.div(cross_table.sum(axis=1), axis=0) * 100
                cross_table_display = cross_table_display.round(2)
            elif display_type == "Pourcentages en colonne":
                cross_table_display = cross_table.div(cross_table.sum(axis=0), axis=1) * 100
                cross_table_display = cross_table_display.round(2)
            elif display_type == "Pourcentages totaux":
                cross_table_display = (cross_table / len(df)) * 100
                cross_table_display = cross_table_display.round(2)
            else:
                cross_table_display = cross_table
            
            st.dataframe(cross_table_display.style.format("{:.2f}" if display_type != "Effectifs" else "{:.0f}"))
            
            # Test du Chi² si demandé
            if show_chi2 and len(cross_table) > 1:
                try:
                    chi2, p_value, dof, expected = stats.chi2_contingency(cross_table)
                    st.write(f"**Test du Chi²:** χ² = {chi2:.3f}, p-value = {p_value:.4f}, ddl = {dof}")
                    
                    if p_value < 0.05:
                        st.success("✅ Association significative (p < 0.05)")
                    else:
                        st.info("ℹ️ Aucune association significative (p ≥ 0.05)")
                except Exception as e:
                    st.warning(f"Test du Chi² non calculable: {str(e)}")
            
            # Téléchargement du tableau
            csv = cross_table_display.to_csv()
            st.download_button(
                label=f"📥 Télécharger {var}",
                data=csv,
                file_name=f"croisement_{var_interet}_{var}.csv",
                mime="text/csv",
                key=f"download_cross_{var}"
            )

def tests_statistiques():
    st.markdown('<h2 class="section-header">📈 Tests Statistiques</h2>', unsafe_allow_html=True)
    
    df = st.session_state.df
    var_interet = st.session_state.var_interet
    
    # Sélection des variables pour le test
    st.markdown("### 🧪 Sélection des Variables pour Test")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_var1 = st.selectbox(
            "Variable 1:",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(var_interet) if var_interet in df.columns else 0
        )
    
    with col2:
        test_var2 = st.selectbox(
            "Variable 2:",
            options=[v for v in df.columns if v != test_var1],
            key="test_var2"
        )
    
    # Types de variables
    var1_type = 'catégorielle' if df[test_var1].dtype == 'object' or df[test_var1].nunique() < 10 else 'numérique'
    var2_type = 'catégorielle' if df[test_var2].dtype == 'object' or df[test_var2].nunique() < 10 else 'numérique'
    
    st.write(f"**Type des variables:** {test_var1} ({var1_type}), {test_var2} ({var2_type})")
    
    # Test automatique selon les types
    if var1_type == 'catégorielle' and var2_type == 'catégorielle':
        st.info("🔍 Test recommandé: Chi-carré d'indépendance")
        test_chi2_carre(df, test_var1, test_var2)
    
    elif var1_type == 'catégorielle' and var2_type == 'numérique':
        st.info("🔍 Test recommandé: ANOVA ou Test-t")
        test_anova_ttest(df, test_var1, test_var2)
    
    elif var1_type == 'numérique' and var2_type == 'numérique':
        st.info("🔍 Test recommandé: Corrélation")
        test_correlation(df, test_var1, test_var2)
    
    else:
        st.warning("Combinaison de types non supportée")

def test_chi2_carre(df, var1, var2):
    """Test du Chi-carré pour deux variables catégorielles"""
    contingency_table = pd.crosstab(df[var1], df[var2])
    
    if contingency_table.size == 0:
        st.error("Tableau de contingence vide")
        return
    
    try:
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Statistique Chi²", f"{chi2:.4f}")
            st.metric("Degrés de liberté", dof)
        with col2:
            st.metric("P-value", f"{p:.4f}")
            if p < 0.05:
                st.success("Association significative (p < 0.05)")
            else:
                st.info("Aucune association significative (p ≥ 0.05)")
        
        # Afficher le tableau de contingence
        st.write("**Tableau de contingence:**")
        st.dataframe(contingency_table)
        
    except Exception as e:
        st.error(f"Erreur dans le test Chi-carré: {str(e)}")

def test_anova_ttest(df, cat_var, num_var):
    """ANOVA ou Test-t pour variable catégorielle vs numérique"""
    groups = df.groupby(cat_var)[num_var].apply(list)
    
    if len(groups) == 2:
        # Test-t pour 2 groupes
        t_stat, p_value = stats.ttest_ind(groups.iloc[0], groups.iloc[1])
        
        st.metric("Statistique t", f"{t_stat:.4f}")
        st.metric("P-value", f"{p_value:.4f}")
        
        if p_value < 0.05:
            st.success("Différence significative entre les groupes (p < 0.05)")
        else:
            st.info("Aucune différence significative entre les groupes (p ≥ 0.05)")
        
        # Afficher les statistiques descriptives par groupe
        st.write("**Statistiques par groupe:**")
        stats_by_group = df.groupby(cat_var)[num_var].agg(['mean', 'std', 'count'])
        st.dataframe(stats_by_group)
        
    else:
        # ANOVA pour plus de 2 groupes
        f_stat, p_value = stats.f_oneway(*groups)
        
        st.metric("Statistique F", f"{f_stat:.4f}")
        st.metric("P-value", f"{p_value:.4f}")
        
        if p_value < 0.05:
            st.success("Différence significative entre les groupes (p < 0.05)")
        else:
            st.info("Aucune différence significative entre les groupes (p ≥ 0.05)")
        
        # Afficher les statistiques descriptives par groupe
        st.write("**Statistiques par groupe:**")
        stats_by_group = df.groupby(cat_var)[num_var].agg(['mean', 'std', 'count'])
        st.dataframe(stats_by_group)

def test_correlation(df, var1, var2):
    """Test de corrélation pour deux variables numériques"""
    # Nettoyer les données
    clean_data = df[[var1, var2]].dropna()
    
    if len(clean_data) < 2:
        st.error("Pas assez de données pour calculer la corrélation")
        return
    
    corr_coef, p_value = stats.pearsonr(clean_data[var1], clean_data[var2])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Coefficient de corrélation", f"{corr_coef:.4f}")
    with col2:
        st.metric("P-value", f"{p_value:.4f}")
    
    # Interprétation
    if abs(corr_coef) > 0.7:
        strength = "forte"
    elif abs(corr_coef) > 0.3:
        strength = "modérée"
    else:
        strength = "faible"
    
    if corr_coef > 0:
        direction = "positive"
    else:
        direction = "négative"
    
    st.write(f"**Interprétation:** {strength} corrélation {direction}")
    
    if p_value < 0.05:
        st.success("Corrélation statistiquement significative (p < 0.05)")
    else:
        st.info("Corrélation non significative (p ≥ 0.05)")
    
    # Graphique de dispersion
    fig = px.scatter(clean_data, x=var1, y=var2, title=f"Relation entre {var1} et {var2}")
    st.plotly_chart(fig, use_container_width=True)

def visualisations():
    st.markdown('<h2 class="section-header">🎨 Visualisations</h2>', unsafe_allow_html=True)
    
    df = st.session_state.df
    var_interet = st.session_state.var_interet
    
    # Sélection du type de graphique
    chart_type = st.selectbox(
        "Type de graphique:",
        ["Diagramme en barres", "Diagramme en bande", "Histogramme", "Boxplot", "Scatter plot"]
    )
    
    # Variables à visualiser
    if chart_type in ["Diagramme en barres", "Diagramme en bande"]:
        x_var = st.selectbox("Variable catégorielle:", df.columns.tolist())
        color_var = st.selectbox("Variable de couleur (optionnel):", [None] + [var_interet] + [v for v in df.columns if v != x_var and v != var_interet])
        
        if chart_type == "Diagramme en barres":
            fig = create_bar_chart(df, x_var, color_var)
        else:
            fig = create_stacked_bar_chart(df, x_var, color_var)
    
    elif chart_type == "Histogramme":
        num_var = st.selectbox("Variable numérique:", 
                              [col for col in df.columns if df[col].dtype in ['int64', 'float64']])
        color_var = st.selectbox("Variable de couleur:", [None] + [var_interet] + [v for v in df.columns if v != num_var and v != var_interet])
        fig = create_histogram(df, num_var, color_var)
    
    elif chart_type == "Boxplot":
        cat_var = st.selectbox("Variable catégorielle:", df.columns.tolist())
        num_var = st.selectbox("Variable numérique:", 
                              [col for col in df.columns if df[col].dtype in ['int64', 'float64']])
        fig = create_boxplot(df, cat_var, num_var)
    
    elif chart_type == "Scatter plot":
        x_var = st.selectbox("Variable X:", 
                            [col for col in df.columns if df[col].dtype in ['int64', 'float64']])
        y_var = st.selectbox("Variable Y:", 
                            [col for col in df.columns if df[col].dtype in ['int64', 'float64']])
        color_var = st.selectbox("Variable de couleur:", [None] + df.columns.tolist())
        fig = create_scatter_plot(df, x_var, y_var, color_var)
    
    # Affichage du graphique
    if 'fig' in locals():
        st.plotly_chart(fig, use_container_width=True)
        
        # Options de téléchargement
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Télécharger le graphique (PNG)"):
                try:
                    fig.write_image("graphique.png")
                    with open("graphique.png", "rb") as file:
                        st.download_button(
                            label="Télécharger PNG",
                            data=file,
                            file_name="graphique.png",
                            mime="image/png"
                        )
                except Exception as e:
                    st.error(f"Erreur lors de l'export PNG: {str(e)}")
        
        with col2:
            # Télécharger les données du graphique
            if 'x_var' in locals() and 'y_var' in locals():
                chart_data = df[[x_var, y_var]].copy()
            elif 'x_var' in locals():
                chart_data = df[[x_var]].copy()
            else:
                chart_data = df[[num_var]].copy()
            
            csv = chart_data.to_csv(index=False)
            st.download_button(
                label="📊 Télécharger les données",
                data=csv,
                file_name="donnees_graphique.csv",
                mime="text/csv"
            )

def create_bar_chart(df, x_var, color_var=None):
    """Crée un diagramme en barres interactif"""
    if color_var:
        fig = px.histogram(df, x=x_var, color=color_var, barmode='group',
                          title=f"Distribution de {x_var} par {color_var}")
    else:
        fig = px.histogram(df, x=x_var, title=f"Distribution de {x_var}")
    
    fig.update_layout(
        xaxis_title=x_var,
        yaxis_title="Effectif",
        legend_title=color_var if color_var else ""
    )
    
    return fig

def create_stacked_bar_chart(df, x_var, stack_var=None):
    """Crée un diagramme en bande (stacked bar chart)"""
    if stack_var:
        fig = px.histogram(df, x=x_var, color=stack_var, barmode='stack',
                          title=f"Diagramme en bande: {x_var} par {stack_var}")
    else:
        fig = px.histogram(df, x=x_var, barmode='stack',
                          title=f"Diagramme en bande: {x_var}")
    
    return fig

def create_histogram(df, num_var, color_var=None):
    """Crée un histogramme"""
    if color_var:
        fig = px.histogram(df, x=num_var, color=color_var, marginal="box",
                          title=f"Distribution de {num_var}")
    else:
        fig = px.histogram(df, x=num_var, title=f"Distribution de {num_var}")
    
    return fig

def create_boxplot(df, cat_var, num_var):
    """Crée un boxplot"""
    fig = px.box(df, x=cat_var, y=num_var, 
                title=f"Distribution de {num_var} par {cat_var}")
    return fig

def create_scatter_plot(df, x_var, y_var, color_var=None):
    """Crée un scatter plot"""
    if color_var:
        fig = px.scatter(df, x=x_var, y=y_var, color=color_var,
                        title=f"Relation entre {x_var} et {y_var}")
    else:
        fig = px.scatter(df, x=x_var, y=y_var,
                        title=f"Relation entre {x_var} et {y_var}")
    
    return fig

def tableaux_3d():
    st.markdown('<h2 class="section-header">📐 Tableaux à Trois Dimensions</h2>', unsafe_allow_html=True)
    
    df = st.session_state.df
    var_interet = st.session_state.var_interet
    
    st.markdown("### 🔮 Tableau Croisé 3D")
    
    # Sélection des 3 variables
    col1, col2, col3 = st.columns(3)
    
    with col1:
        var1 = st.selectbox("Variable ligne:", df.columns.tolist(), key="3d_var1")
    
    with col2:
        var2 = st.selectbox("Variable colonne:", 
                           [v for v in df.columns if v != var1], 
                           key="3d_var2")
    
    with col3:
        var3 = st.selectbox("Variable profondeur:", 
                           [v for v in df.columns if v not in [var1, var2]], 
                           key="3d_var3")
    
    # Type d'agrégation
    agg_type = st.selectbox("Type d'agrégation:", 
                           ["Effectif", "Moyenne", "Somme", "Pourcentage"])
    
    if st.button("🔄 Générer le tableau 3D"):
        # Création du tableau 3D
        if agg_type == "Effectif":
            pivot_3d = df.pivot_table(
                index=var1, 
                columns=[var2, var3], 
                aggfunc='size',
                fill_value=0
            )
        elif agg_type in ["Moyenne", "Somme"]:
            # Nécessite une variable numérique
            num_vars = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
            if num_vars:
                num_var = st.selectbox("Variable à agréger:", num_vars, key="agg_var")
                agg_func = 'mean' if agg_type == "Moyenne" else 'sum'
                pivot_3d = df.pivot_table(
                    index=var1, 
                    columns=[var2, var3], 
                    values=num_var,
                    aggfunc=agg_func,
                    fill_value=0
                )
            else:
                st.warning("Aucune variable numérique disponible")
                return
        else:  # Pourcentage
            pivot_3d = df.pivot_table(
                index=var1, 
                columns=[var2, var3], 
                aggfunc='size',
                fill_value=0
            )
            pivot_3d = (pivot_3d / pivot_3d.sum().sum()) * 100
        
        # Affichage du tableau
        st.dataframe(pivot_3d.style.background_gradient(cmap='Blues'))
        
        # Téléchargement du tableau 3D
        csv = pivot_3d.to_csv()
        st.download_button(
            label="📥 Télécharger le tableau 3D",
            data=csv,
            file_name=f"tableau_3d_{var1}_{var2}_{var3}.csv",
            mime="text/csv"
        )

def main():
    # Charger le CSS personnalisé
    load_css()
    
    # Ajouter le logo dans la sidebar
    add_logo()
    
    # Sidebar pour la navigation
    st.sidebar.markdown("## 📋 Navigation")
    section = st.sidebar.radio(
        "Sélectionnez une section:",
        ["🏠 Accueil", "📥 Télécharger données", "📁 Chargement des données", "📊 Répartition des variables", 
         "🔍 Tableaux croisés", "📈 Tests statistiques", "🎨 Visualisations", "📐 Tableaux 3D"]
    )
    
    # Informations utilisateur dans la sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 Session Utilisateur")
    st.sidebar.info("""
    **Statut:** Connecté  
    **Type:** Analyste  
    **Version:** Pro 2.0
    """)
    
    # Initialisation des variables de session
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'var_interet' not in st.session_state:
        st.session_state.var_interet = None
    if 'var_independantes' not in st.session_state:
        st.session_state.var_independantes = []
    if 'generated_data' not in st.session_state:
        st.session_state.generated_data = None
    
    # Page d'accueil
    if section == "🏠 Accueil":
        show_welcome()
    
    # Section 1: Téléchargement de données
    elif section == "📥 Télécharger données":
        st.markdown('<h2 class="section-header">📥 Télécharger des Données d\'Exemple</h2>', unsafe_allow_html=True)
        telecharger_donnees()
    
    # Section 2: Chargement des données
    elif section == "📁 Chargement des données":
        st.markdown('<h2 class="section-header">📁 Chargement des Données</h2>', unsafe_allow_html=True)
        charger_donnees()
    
    # Sections suivantes seulement si des données sont chargées
    elif st.session_state.df is not None:
        if section == "📊 Répartition des variables":
            st.markdown('<h2 class="section-header">📊 Répartition de Toutes les Variables</h2>', unsafe_allow_html=True)
            repartition_variables()
        elif section == "🔍 Tableaux croisés":
            st.markdown('<h2 class="section-header">🔍 Tableaux Croisés</h2>', unsafe_allow_html=True)
            tableaux_croises()
        elif section == "📈 Tests statistiques":
            st.markdown('<h2 class="section-header">📈 Tests Statistiques</h2>', unsafe_allow_html=True)
            tests_statistiques()
        elif section == "🎨 Visualisations":
            st.markdown('<h2 class="section-header">🎨 Visualisations</h2>', unsafe_allow_html=True)
            visualisations()
        elif section == "📐 Tableaux 3D":
            st.markdown('<h2 class="section-header">📐 Tableaux à Trois Dimensions</h2>', unsafe_allow_html=True)
            tableaux_3d()
    else:
        st.markdown("""
        <div class="warning-message">
            <h4>⚠️ Données Requises</h4>
            <p>Veuillez d'abord charger des données dans la section <strong>'Chargement des données'</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Ajouter le footer sur toutes les pages
    add_footer()

if __name__ == "__main__":
    main()