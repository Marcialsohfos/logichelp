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
import re

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
    def generate_frequency_table(df, variable, group_variable, max_categories=15):
        """Fallback function"""
        try:
            cross_tab = pd.crosstab(df[variable], df[group_variable], margins=True)
            return cross_tab
        except:
            return pd.DataFrame({"Info": ["Analyse non disponible"]})
    
    class DataGenerator:
        def generate_complex_dataset(self, **kwargs):
            st.error("Générateur de données non disponible")
            return pd.DataFrame()

try:
    from templates.download_pages import show_download_section, show_data_preview, show_data_quality_report
except ImportError:
    def show_download_section(generated_data):
        from data_downloader import DataDownloader
        downloader = DataDownloader()
        st.markdown("### 💾 Télécharger les Données")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📊 Excel"):
                downloader.download_excel(generated_data, "donnees_analyse.xlsx")
        with col2:
            if st.button("📝 CSV"):
                downloader.download_csv(generated_data, "donnees_analyse.csv")
        with col3:
            if st.button("🎯 STATA"):
                downloader.download_stata(generated_data, "donnees_analyse.dta")
    
    def show_data_preview(generated_data):
        if generated_data is not None:
            st.dataframe(generated_data.head(), width='stretch')
            st.write(f"**Dimensions:** {generated_data.shape[0]} lignes × {generated_data.shape[1]} colonnes")
    
    def show_data_quality_report(generated_data):
        if generated_data is not None:
            completeness = (1 - generated_data.isna().sum().sum() / (generated_data.shape[0] * generated_data.shape[1])) * 100
            st.metric("Complétude des données", f"{completeness:.1f}%")

from data_downloader import DataDownloader

# Configuration de la page
st.set_page_config(
    page_title="LogicApp Analytics Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonctions de nettoyage pour Excel
def clean_sheet_name(name):
    """
    Nettoie le nom de la feuille pour être compatible Excel
    """
    cleaned = re.sub(r'[\\/*?\[\]]', '_', str(name))
    return cleaned[:31]

def clean_filename(name):
    """
    Nettoie le nom de fichier
    """
    cleaned = re.sub(r'[\\/*?\[\]:]', '_', str(name))
    return cleaned

# CSS personnalisé avec design professionnel
def load_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.8rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
            font-weight: 700;
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
        .stButton button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 500;
            width: 100%;
        }
        .download-btn {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
        }
        .footer {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 1.5rem;
            text-align: center;
            margin-top: 3rem;
            border-radius: 8px;
        }
        .warning-box {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .success-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .export-section {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border: 2px solid #667eea;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

def add_footer():
    st.markdown("""
    <div class="footer">
        <div style="display: flex; justify-content: space-between; align-items: center; max-width: 1200px; margin: 0 auto;">
            <div>
                <strong>LogicApp Analytics Pro</strong> - Plateforme d'Analyse Scientifique
            </div>
            <div>
                🔬 Powered by <strong>LogicApp Analytics</strong>
            </div>
            <div>
                © Copyright 2025 - Tous droits réservés
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def add_logo():
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 8px; margin-bottom: 1rem;">
        <h2 style="color: white; margin-bottom: 0.5rem;">🔬</h2>
        <h3 style="color: white; margin: 0;">LogicApp Analytics</h3>
        <p style="color: #e0e0e0; font-size: 0.8rem; margin: 0;">Pro Edition</p>
    </div>
    """, unsafe_allow_html=True)

def show_welcome():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h1 class="main-header">🔬 LogicApp Analytics Pro</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #6c757d; font-size: 1.2rem;">Plateforme Professionnelle d\'Analyse de Données Scientifiques</p>', unsafe_allow_html=True)
        
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

def generate_frequency_table(df, variable, group_variable, max_categories=15):
    """
    Version robuste de génération de tableaux de fréquences
    Avec gestion améliorée des types de données
    """
    try:
        # Vérifications de base
        if variable not in df.columns or group_variable not in df.columns:
            return pd.DataFrame({"Erreur": ["Colonne manquante"]})
        
        # Nettoyer les données - conversion explicite en string
        df_clean = df[[variable, group_variable]].dropna().copy()
        df_clean[variable] = df_clean[variable].astype(str)
        df_clean[group_variable] = df_clean[group_variable].astype(str)
        
        if df_clean.empty:
            return pd.DataFrame({"Message": ["Aucune donnée après nettoyage"]})
        
        # Vérifier la variabilité
        if df_clean[variable].nunique() <= 1 or df_clean[group_variable].nunique() <= 1:
            return pd.DataFrame({"Message": ["Pas assez de variabilité dans les données"]})
        
        # Limiter les catégories si nécessaire
        if df_clean[variable].nunique() > max_categories:
            top_categories = df_clean[variable].value_counts().head(max_categories - 1).index
            df_clean[variable] = df_clean[variable].apply(
                lambda x: x if x in top_categories else 'Autres'
            )
        
        # Créer le tableau croisé avec gestion d'erreur
        try:
            cross_tab = pd.crosstab(
                df_clean[variable], 
                df_clean[group_variable],
                margins=True,
                margins_name="Total"
            )
        except Exception as cross_error:
            return pd.DataFrame({"Erreur": [f"Erreur création tableau: {str(cross_error)}"]})
        
        # Vérifier que le tableau n'est pas vide
        if cross_tab.empty:
            return pd.DataFrame({"Message": ["Tableau croisé vide"]})
        
        # Calculer les pourcentages de manière sécurisée
        try:
            # Utiliser sum() pour plus de sécurité
            total_values = cross_tab.sum(axis=0)
            percent_tab = (cross_tab / total_values) * 100
            
            # Formater le résultat avec gestion des erreurs
            result_data = {}
            for col in cross_tab.columns:
                result_data[col] = [
                    f"{count} ({percent:.1f}%)" 
                    for count, percent in zip(cross_tab[col], percent_tab[col])
                ]
            
            result_df = pd.DataFrame(result_data, index=cross_tab.index.astype(str))
            return result_df
            
        except Exception as calc_error:
            # Fallback: seulement les effectifs avec index string
            cross_tab.index = cross_tab.index.astype(str)
            return cross_tab
            
    except Exception as e:
        error_msg = f"Erreur avec {variable}: {str(e)}"
        return pd.DataFrame({"Erreur": [error_msg]})

def download_all_tables_excel(all_tables_data, var_interet):
    """
    Télécharge tous les tableaux de répartition dans un seul fichier Excel
    Version corrigée des erreurs de caractères et de conversion
    """
    try:
        # Créer un fichier Excel en mémoire
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Feuille de sommaire
            summary_data = {
                'Variable': [],
                'Nombre de Catégories': [],
                'Dimensions du Tableau': [],
                'Date de Génération': []
            }
            
            for variable_name, table in all_tables_data.items():
                # Nettoyer le nom de la feuille pour Excel
                sheet_name = clean_sheet_name(variable_name)
                
                # S'assurer que l'index est de type string pour éviter les erreurs de conversion
                table_display = table.copy()
                table_display.index = table_display.index.astype(str)
                table_display.columns = table_display.columns.astype(str)
                
                # Ajouter chaque tableau dans une feuille séparée
                table_display.to_excel(writer, sheet_name=sheet_name, index=True)
                
                # Remplir le sommaire
                summary_data['Variable'].append(variable_name)
                summary_data['Nombre de Catégories'].append(len(table.index))
                summary_data['Dimensions du Tableau'].append(f"{table.shape[0]}x{table.shape[1]}")
                summary_data['Date de Génération'].append(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
            
            # Créer la feuille de sommaire
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='SOMMAIRE', index=False)
            
            # Feuille de métadonnées
            metadata_data = {
                'Information': [
                    'Nom du Fichier',
                    'Variable d\'Intérêt',
                    'Nombre de Variables Analysées',
                    'Date de Création',
                    'Généré par'
                ],
                'Valeur': [
                    f'repartition_complete_{clean_filename(var_interet)}.xlsx',
                    var_interet,
                    len(all_tables_data),
                    pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'LogicApp Analytics Pro'
                ]
            }
            metadata_df = pd.DataFrame(metadata_data)
            metadata_df.to_excel(writer, sheet_name='METADONNEES', index=False)
            
            # Feuille avec les instructions
            instructions_data = {
                'Section': [
                    'SOMMAIRE',
                    'METADONNEES',
                    'Feuilles des Variables'
                ],
                'Description': [
                    'Vue d\'ensemble de toutes les variables analysées',
                    'Informations techniques sur le fichier',
                    'Une feuille par variable avec les répartitions détaillées'
                ],
                'Utilisation': [
                    'Identifier rapidement les variables disponibles',
                    'Comprendre le contexte de l\'analyse',
                    'Analyser les répartitions détaillées par variable'
                ]
            }
            instructions_df = pd.DataFrame(instructions_data)
            instructions_df.to_excel(writer, sheet_name='INSTRUCTIONS', index=False)
        
        output.seek(0)
        
        # Télécharger le fichier
        filename = f"repartition_complete_{clean_filename(var_interet)}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx"
        
        st.download_button(
            label="📥 Cliquez pour télécharger le fichier Excel complet",
            data=output.getvalue(),
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"download_all_tables_excel_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
        )
        
        st.success(f"✅ Fichier Excel généré avec succès! ({len(all_tables_data)} variables)")
        
    except Exception as e:
        st.error(f"❌ Erreur lors de la génération du fichier Excel: {str(e)}")
        # Log détaillé pour le débogage
        st.error(f"Détails de l'erreur: {type(e).__name__}")

def telecharger_donnees():
    st.markdown('<h2 class="section-header">📥 Télécharger des Données d\'Exemple</h2>', unsafe_allow_html=True)
    
    st.info("""
    **📋 Instructions :**
    - Téléchargez un jeu de données d'exemple dans le format de votre choix
    - Utilisez ces données pour tester l'application
    - Les données contiennent des variables catégorielles et numériques réalistes
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        n_observations = st.slider("Nombre d'observations", 100, 5000, 1000)
        n_variables = st.slider("Nombre de variables", 5, 30, 15)
    with col2:
        include_missing = st.checkbox("Inclure des valeurs manquantes", value=True)
        missing_percentage = st.slider("Pourcentage de valeurs manquantes", 0.0, 20.0, 5.0) if include_missing else 0.0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        n_categorical = st.number_input("Variables catégorielles", 1, n_variables-2, 5)
    with col2:
        n_numerical = st.number_input("Variables numériques", 1, n_variables-2, 7)
    with col3:
        n_binary = st.number_input("Variables binaires", 1, n_variables-2, 3)
    
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
    
    if st.session_state.generated_data is not None:
        show_data_preview(st.session_state.generated_data)
        show_data_quality_report(st.session_state.generated_data)
        show_download_section(st.session_state.generated_data)

def charger_donnees():
    st.markdown('<h2 class="section-header">📁 Chargement des Données</h2>', unsafe_allow_html=True)
    
    downloader = DataDownloader()
    
    data_source = st.radio(
        "Source des données:",
        ["📤 Uploader un fichier", "🎯 Utiliser les données générées"]
    )
    
    if data_source == "📤 Uploader un fichier":
        st.info("💡 Formats supportés: Excel (.xlsx, .xls), CSV, STATA, TXT")
        df = downloader.upload_data()
        
        if df is not None:
            st.session_state.df = df
            downloader.show_file_preview(df)
    
    else:
        if st.session_state.generated_data is not None:
            st.session_state.df = st.session_state.generated_data
            st.success("✅ Données générées chargées avec succès!")
            downloader.show_file_preview(st.session_state.df)
        else:
            st.warning("⚠️ Veuillez d'abord générer des données dans la section 'Télécharger données'")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
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
    
    # Vérifications préalables
    if var_interet not in df.columns:
        st.error(f"❌ Variable d'intérêt '{var_interet}' non trouvée")
        return
    
    # Options d'affichage
    col1, col2 = st.columns(2)
    with col1:
        max_categories = st.number_input("Max catégories par variable", min_value=5, max_value=50, value=15)
    with col2:
        min_unique_values = st.number_input("Valeurs uniques minimum", min_value=1, max_value=10, value=2)
    
    # Filtrer les variables valides
    valid_variables = []
    problematic_variables = []
    
    for col in df.columns:
        if col != var_interet:
            try:
                unique_vals_var = df[col].nunique()
                unique_vals_target = df[var_interet].nunique()
                
                if (unique_vals_var >= min_unique_values and 
                    unique_vals_target >= min_unique_values and
                    not df[col].isna().all() and 
                    not df[var_interet].isna().all()):
                    valid_variables.append(col)
                else:
                    problematic_variables.append((col, unique_vals_var, unique_vals_target))
            except:
                problematic_variables.append((col, "Erreur", "Erreur"))
    
    # Afficher les variables problématiques
    if problematic_variables:
        with st.expander("⚠️ Variables ignorées (cliquer pour voir)"):
            for var, unique_var, unique_target in problematic_variables:
                st.write(f"- **{var}**: {unique_var} valeur(s) unique(s) | Variable cible: {unique_target} valeur(s) unique(s)")
    
    if not valid_variables:
        st.error("❌ Aucune variable valide à analyser. Vérifiez que vos données ont suffisamment de variabilité.")
        return
    
    st.info(f"🔍 Analyse de {len(valid_variables)} variables sur {len(df.columns) - 1} totales")
    
    # Génération des tableaux
    progress_bar = st.progress(0)
    successful_tables = 0
    all_tables_data = {}  # Stocker tous les tableaux pour l'export Excel
    
    for i, variable in enumerate(valid_variables):
        try:
            with st.spinner(f"Analyse de {variable}..."):
                table = generate_frequency_table(df, variable, var_interet, max_categories)
                
                if table is not None and not table.empty:
                    if 'Erreur' not in table.columns and 'Message' not in table.columns:
                        successful_tables += 1
                        # Stocker le tableau pour l'export Excel
                        all_tables_data[variable] = table
                        
                        with st.expander(f"📋 {variable} ({df[variable].nunique()} catégories)", expanded=False):
                            st.dataframe(table, width='stretch')
                            
                            # Téléchargement individuel
                            try:
                                csv = table.to_csv()
                                st.download_button(
                                    label=f"📥 Télécharger {variable} (CSV)",
                                    data=csv,
                                    file_name=f"repartition_{clean_filename(variable)}.csv",
                                    mime="text/csv",
                                    key=f"dl_{variable}_{i}"
                                )
                            except Exception as e:
                                st.error(f"❌ Export impossible: {str(e)}")
                    else:
                        # Afficher les messages d'erreur
                        with st.expander(f"❌ {variable} - Problème", expanded=False):
                            st.dataframe(table, width='stretch')
                else:
                    st.warning(f"⚠️ Tableau vide pour {variable}")
                    
        except Exception as e:
            st.error(f"❌ Erreur critique avec {variable}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(valid_variables))
    
    # SECTION TÉLÉCHARGEMENT GLOBAL - NOUVELLE FONCTIONNALITÉ
    if successful_tables > 0:
        st.success(f"✅ {successful_tables}/{len(valid_variables)} tableaux générés avec succès")
        
        st.markdown("---")
        st.markdown("### 💾 Téléchargement Global des Résultats")
        
        # Section d'export avec style
        st.markdown("""
        <div class="export-section">
            <h4>📊 Export Complet en Excel</h4>
            <p>Exportez tous les tableaux de répartition dans un seul fichier Excel organisé</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("""
            **Le fichier Excel contiendra:**
            • Une feuille par variable avec son tableau complet
            • Un sommaire avec toutes les statistiques
            • Des métadonnées techniques
            • Un guide d'utilisation
            """)
        
        with col2:
            if st.button("🚀 Générer le fichier Excel complet", type="primary", use_container_width=True):
                download_all_tables_excel(all_tables_data, var_interet)
        
        # Statistiques de l'export
        st.write(f"**📁 Contenu du fichier:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Feuilles de données", len(all_tables_data))
        with col2:
            st.metric("Variables analysées", successful_tables)
        with col3:
            st.metric("Feuilles totales", len(all_tables_data) + 3)  # +3 pour sommaire, métadonnées, instructions
        
    else:
        st.warning("⚠️ Aucun tableau n'a pu être généré. Vérifiez vos données.")

def tableaux_croises():
    st.markdown('<h2 class="section-header">🔍 Tableaux Croisés</h2>', unsafe_allow_html=True)
    
    df = st.session_state.df
    var_interet = st.session_state.var_interet
    var_independantes = st.session_state.var_independantes
    
    if not var_independantes:
        st.warning("Veuillez sélectionner des variables indépendantes")
        return
    
    selected_vars = st.multiselect(
        "Choisir les variables à croiser avec la variable d'intérêt:",
        options=var_independantes,
        default=var_independantes[:2] if len(var_independantes) >= 2 else var_independantes
    )
    
    if selected_vars:
        col1, col2 = st.columns(2)
        with col1:
            display_type = st.radio(
                "Type d'affichage:",
                ["Effectifs", "Pourcentages en ligne", "Pourcentages en colonne", "Pourcentages totaux"]
            )
        with col2:
            show_chi2 = st.checkbox("Afficher test du Chi²", value=True)
        
        for var in selected_vars:
            st.markdown(f"### 📊 Croisement: {var_interet} × {var}")
            
            try:
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
                
                st.dataframe(cross_table_display.style.format("{:.2f}" if display_type != "Effectifs" else "{:.0f}"), width='stretch')
                
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
                
                csv = cross_table_display.to_csv()
                st.download_button(
                    label=f"📥 Télécharger {var}",
                    data=csv,
                    file_name=f"croisement_{var_interet}_{var}.csv",
                    mime="text/csv",
                    key=f"download_cross_{var}"
                )
                
            except Exception as e:
                st.error(f"❌ Erreur avec la variable {var}: {str(e)}")

def tests_statistiques():
    st.markdown('<h2 class="section-header">📈 Tests Statistiques</h2>', unsafe_allow_html=True)
    
    df = st.session_state.df
    var_interet = st.session_state.var_interet
    
    st.markdown("### 🧪 Sélection des Variables pour Test")
    
    col1, col2 = st.columns(2)
    with col1:
        test_var1 = st.selectbox("Variable 1:", options=df.columns.tolist(),
                               index=df.columns.tolist().index(var_interet) if var_interet in df.columns else 0)
    with col2:
        test_var2 = st.selectbox("Variable 2:", options=[v for v in df.columns if v != test_var1], key="test_var2")
    
    # Déterminer les types de variables
    try:
        var1_type = 'catégorielle' if df[test_var1].dtype == 'object' or df[test_var1].nunique() < 10 else 'numérique'
        var2_type = 'catégorielle' if df[test_var2].dtype == 'object' or df[test_var2].nunique() < 10 else 'numérique'
        
        st.write(f"**Type des variables:** {test_var1} ({var1_type}), {test_var2} ({var2_type})")
        
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
    except Exception as e:
        st.error(f"❌ Erreur lors de l'analyse des types: {str(e)}")

def test_chi2_carre(df, var1, var2):
    try:
        contingency_table = pd.crosstab(df[var1], df[var2])
        if contingency_table.size == 0:
            st.error("Tableau de contingence vide")
            return
        
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
        st.write("**Tableau de contingence:**")
        st.dataframe(contingency_table, width='stretch')
    except Exception as e:
        st.error(f"Erreur dans le test Chi-carré: {str(e)}")

def test_anova_ttest(df, cat_var, num_var):
    try:
        groups = df.groupby(cat_var)[num_var].apply(list)
        if len(groups) == 2:
            t_stat, p_value = stats.ttest_ind(groups.iloc[0], groups.iloc[1])
            st.metric("Statistique t", f"{t_stat:.4f}")
            st.metric("P-value", f"{p_value:.4f}")
            if p_value < 0.05:
                st.success("Différence significative entre les groupes (p < 0.05)")
            else:
                st.info("Aucune différence significative entre les groupes (p ≥ 0.05)")
            st.write("**Statistiques par groupe:**")
            stats_by_group = df.groupby(cat_var)[num_var].agg(['mean', 'std', 'count'])
            st.dataframe(stats_by_group, width='stretch')
        else:
            f_stat, p_value = stats.f_oneway(*groups)
            st.metric("Statistique F", f"{f_stat:.4f}")
            st.metric("P-value", f"{p_value:.4f}")
            if p_value < 0.05:
                st.success("Différence significative entre les groupes (p < 0.05)")
            else:
                st.info("Aucune différence significative entre les groupes (p ≥ 0.05)")
            st.write("**Statistiques par groupe:**")
            stats_by_group = df.groupby(cat_var)[num_var].agg(['mean', 'std', 'count'])
            st.dataframe(stats_by_group, width='stretch')
    except Exception as e:
        st.error(f"Erreur dans le test ANOVA/Test-t: {str(e)}")

def test_correlation(df, var1, var2):
    try:
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
        if abs(corr_coef) > 0.7:
            strength = "forte"
        elif abs(corr_coef) > 0.3:
            strength = "modérée"
        else:
            strength = "faible"
        direction = "positive" if corr_coef > 0 else "négative"
        st.write(f"**Interprétation:** {strength} corrélation {direction}")
        if p_value < 0.05:
            st.success("Corrélation statistiquement significative (p < 0.05)")
        else:
            st.info("Corrélation non significative (p ≥ 0.05)")
        fig = px.scatter(clean_data, x=var1, y=var2, title=f"Relation entre {var1} et {var2}")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur dans le test de corrélation: {str(e)}")

def visualisations():
    st.markdown('<h2 class="section-header">🎨 Visualisations</h2>', unsafe_allow_html=True)
    
    df = st.session_state.df
    var_interet = st.session_state.var_interet
    
    chart_type = st.selectbox("Type de graphique:", ["Diagramme en barres", "Diagramme en bande", "Histogramme", "Boxplot", "Scatter plot"])
    
    try:
        if chart_type in ["Diagramme en barres", "Diagramme en bande"]:
            x_var = st.selectbox("Variable catégorielle:", df.columns.tolist())
            color_var = st.selectbox("Variable de couleur:", [None] + [var_interet] + [v for v in df.columns if v != x_var and v != var_interet])
            if chart_type == "Diagramme en barres":
                fig = px.histogram(df, x=x_var, color=color_var, barmode='group', title=f"Distribution de {x_var} par {color_var}")
            else:
                fig = px.histogram(df, x=x_var, color=color_var, barmode='stack', title=f"Diagramme en bande: {x_var} par {color_var}")
        
        elif chart_type == "Histogramme":
            num_var = st.selectbox("Variable numérique:", [col for col in df.columns if df[col].dtype in ['int64', 'float64']])
            color_var = st.selectbox("Variable de couleur:", [None] + [var_interet] + [v for v in df.columns if v != num_var and v != var_interet])
            fig = px.histogram(df, x=num_var, color=color_var, marginal="box", title=f"Distribution de {num_var}")
        
        elif chart_type == "Boxplot":
            cat_var = st.selectbox("Variable catégorielle:", df.columns.tolist())
            num_var = st.selectbox("Variable numérique:", [col for col in df.columns if df[col].dtype in ['int64', 'float64']])
            fig = px.box(df, x=cat_var, y=num_var, title=f"Distribution de {num_var} par {cat_var}")
        
        elif chart_type == "Scatter plot":
            x_var = st.selectbox("Variable X:", [col for col in df.columns if df[col].dtype in ['int64', 'float64']])
            y_var = st.selectbox("Variable Y:", [col for col in df.columns if df[col].dtype in ['int64', 'float64']])
            color_var = st.selectbox("Variable de couleur:", [None] + df.columns.tolist())
            fig = px.scatter(df, x=x_var, y=y_var, color=color_var, title=f"Relation entre {x_var} et {y_var}")
        
        if 'fig' in locals():
            st.plotly_chart(fig, use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                try:
                    if st.button("📥 Télécharger le graphique (PNG)"):
                        fig.write_image("graphique.png")
                        with open("graphique.png", "rb") as file:
                            st.download_button("Télécharger PNG", data=file, file_name="graphique.png", mime="image/png")
                except: 
                    st.info("❌ L'export PNG n'est pas disponible sur cette plateforme")
            with col2:
                if 'x_var' in locals() and 'y_var' in locals():
                    chart_data = df[[x_var, y_var]].copy()
                elif 'x_var' in locals():
                    chart_data = df[[x_var]].copy()
                else:
                    chart_data = df[[num_var]].copy()
                csv = chart_data.to_csv(index=False)
                st.download_button("📊 Télécharger les données", data=csv, file_name="donnees_graphique.csv", mime="text/csv")
                
    except Exception as e:
        st.error(f"❌ Erreur lors de la création du graphique: {str(e)}")

def tableaux_3d():
    st.markdown('<h2 class="section-header">📐 Tableaux à Trois Dimensions</h2>', unsafe_allow_html=True)
    
    df = st.session_state.df
    var_interet = st.session_state.var_interet
    
    st.markdown("### 🔮 Tableau Croisé 3D")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        var1 = st.selectbox("Variable ligne:", df.columns.tolist(), key="3d_var1")
    with col2:
        var2 = st.selectbox("Variable colonne:", [v for v in df.columns if v != var1], key="3d_var2")
    with col3:
        var3 = st.selectbox("Variable profondeur:", [v for v in df.columns if v not in [var1, var2]], key="3d_var3")
    
    agg_type = st.selectbox("Type d'agrégation:", ["Effectif", "Moyenne", "Somme", "Pourcentage"])
    
    if st.button("🔄 Générer le tableau 3D"):
        try:
            if agg_type == "Effectif":
                pivot_3d = df.pivot_table(index=var1, columns=[var2, var3], aggfunc='size', fill_value=0)
            elif agg_type in ["Moyenne", "Somme"]:
                num_vars = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
                if num_vars:
                    num_var = st.selectbox("Variable à agréger:", num_vars, key="agg_var")
                    agg_func = 'mean' if agg_type == "Moyenne" else 'sum'
                    pivot_3d = df.pivot_table(index=var1, columns=[var2, var3], values=num_var, aggfunc=agg_func, fill_value=0)
                else:
                    st.warning("Aucune variable numérique disponible")
                    return
            else:
                pivot_3d = df.pivot_table(index=var1, columns=[var2, var3], aggfunc='size', fill_value=0)
                pivot_3d = (pivot_3d / pivot_3d.sum().sum()) * 100
            
            st.dataframe(pivot_3d.style.background_gradient(cmap='Blues'), width='stretch')
            csv = pivot_3d.to_csv()
            st.download_button("📥 Télécharger le tableau 3D", data=csv, file_name=f"tableau_3d_{var1}_{var2}_{var3}.csv", mime="text/csv")
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la création du tableau 3D: {str(e)}")

def main():
    load_css()
    add_logo()
    
    st.sidebar.markdown("## 📋 Navigation")
    section = st.sidebar.radio("Sélectionnez une section:", ["🏠 Accueil", "📥 Télécharger données", "📁 Chargement des données", "📊 Répartition des variables", "🔍 Tableaux croisés", "📈 Tests statistiques", "🎨 Visualisations", "📐 Tableaux 3D"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 Session Utilisateur")
    st.sidebar.info("**Statut:** Connecté  \n**Type:** Analyste  \n**Version:** Pro 2.0")
    
    # Initialisation des variables de session
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'var_interet' not in st.session_state:
        st.session_state.var_interet = None
    if 'var_independantes' not in st.session_state:
        st.session_state.var_independantes = []
    if 'generated_data' not in st.session_state:
        st.session_state.generated_data = None
    
    if section == "🏠 Accueil":
        show_welcome()
    elif section == "📥 Télécharger données":
        telecharger_donnees()
    elif section == "📁 Chargement des données":
        charger_donnees()
    elif st.session_state.df is not None:
        if section == "📊 Répartition des variables":
            repartition_variables()
        elif section == "🔍 Tableaux croisés":
            tableaux_croises()
        elif section == "📈 Tests statistiques":
            tests_statistiques()
        elif section == "🎨 Visualisations":
            visualisations()
        elif section == "📐 Tableaux 3D":
            tableaux_3d()
    else:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Données Requises</h4>
            <p>Veuillez d'abord charger des données dans la section <strong>'Chargement des données'</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    add_footer()

if __name__ == "__main__":
    main()