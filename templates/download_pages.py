import streamlit as st
import pandas as pd
import io
from data_downloader import DataDownloader

def show_download_section(generated_data):
    """
    Affiche la section de téléchargement avec interface utilisateur
    """
    st.markdown("### 💾 Télécharger les Données")
    
    if generated_data is None:
        st.warning("Aucune donnée disponible pour le téléchargement.")
        return
    
    downloader = DataDownloader()
    
    # Section de téléchargement rapide
    st.markdown("#### 🚀 Téléchargement Rapide")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Excel", key="excel_quick"):
            downloader.download_excel(generated_data, "donnees_analyse.xlsx")
    
    with col2:
        if st.button("📝 CSV", key="csv_quick"):
            downloader.download_csv(generated_data, "donnees_analyse.csv")
    
    with col3:
        if st.button("🎯 STATA", key="stata_quick"):
            downloader.download_stata(generated_data, "donnees_analyse.dta")
    
    with col4:
        if st.button("📋 SPSS", key="spss_quick"):
            downloader.download_spss(generated_data, "donnees_analyse.sav")
    
    # Section de téléchargement personnalisé
    st.markdown("#### 🛠️ Téléchargement Personnalisé")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        custom_format = st.selectbox(
            "Format de fichier:",
            ["Excel", "CSV", "STATA", "SPSS", "TXT"],
            key="custom_format"
        )
    
    with col2:
        custom_filename = st.text_input(
            "Nom du fichier:",
            "mes_donnees_analyse",
            key="custom_filename"
        )
    
    with col3:
        st.write("")  # Espacement
        st.write("")  # Espacement
        download_custom = st.button("🚀 Télécharger", key="custom_download")
    
    if download_custom:
        file_extension = {
            "Excel": ".xlsx",
            "CSV": ".csv", 
            "STATA": ".dta",
            "SPSS": ".sav",
            "TXT": ".txt"
        }[custom_format]
        
        filename = f"{custom_filename}{file_extension}"
        
        if custom_format == "Excel":
            downloader.download_excel(generated_data, filename)
        elif custom_format == "CSV":
            downloader.download_csv(generated_data, filename)
        elif custom_format == "STATA":
            downloader.download_stata(generated_data, filename)
        elif custom_format == "SPSS":
            downloader.download_spss(generated_data, filename)
        elif custom_format == "TXT":
            downloader.download_txt(generated_data, filename)
    
    # Section d'export avancé
    st.markdown("#### ⚙️ Export Avancé")
    
    with st.expander("Options d'export avancées"):
        col1, col2 = st.columns(2)
        
        with col1:
            include_index = st.checkbox("Inclure l'index", value=False)
            encoding = st.selectbox("Encodage:", ["utf-8", "latin-1", "windows-1252"])
        
        with col2:
            if custom_format == "CSV" or custom_format == "TXT":
                delimiter = st.selectbox("Séparateur:", [",", ";", "\t", "|"])
            else:
                delimiter = ","
        
        # Export avec options avancées
        if st.button("🔄 Régénérer avec options", key="advanced_export"):
            st.info("Utilisez les options sélectionnées pour le prochain téléchargement")

def show_data_preview(generated_data):
    """
    Affiche un aperçu des données avec statistiques
    """
    if generated_data is None:
        return
    
    st.markdown("### 👀 Aperçu des Données Générées")
    
    # Onglets pour différentes vues
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Données", "📊 Statistiques", "🔍 Types", "📈 Visualisation rapide"])
    
    with tab1:
        st.dataframe(generated_data.head(10), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Dimensions:** {generated_data.shape[0]} lignes × {generated_data.shape[1]} colonnes")
        with col2:
            missing_total = generated_data.isna().sum().sum()
            st.write(f"**Valeurs manquantes:** {missing_total}")
    
    with tab2:
        st.write("**Statistiques descriptives (variables numériques):**")
        numerical_cols = generated_data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            st.dataframe(generated_data[numerical_cols].describe())
        else:
            st.info("Aucune variable numérique trouvée")
        
        st.write("**Statistiques (variables catégorielles):**")
        categorical_cols = generated_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            st.write(f"**{col}:** {generated_data[col].nunique()} catégories")
            if generated_data[col].nunique() <= 10:
                st.write(f"  - {dict(generated_data[col].value_counts().head())}")
    
    with tab3:
        st.write("**Types de données:**")
        type_summary = generated_data.dtypes.value_counts()
        for dtype, count in type_summary.items():
            st.write(f"- **{dtype}:** {count} colonne(s)")
        
        st.write("**Liste des colonnes:**")
        for col in generated_data.columns:
            dtype = generated_data[col].dtype
            unique_vals = generated_data[col].nunique()
            missing_vals = generated_data[col].isna().sum()
            st.write(f"- **{col}** ({dtype}) - {unique_vals} valeurs uniques - {missing_vals} manquantes")
    
    with tab4:
        st.write("**Visualisation rapide de la distribution:**")
        
        # Sélectionner une colonne pour la visualisation
        viz_col = st.selectbox(
            "Choisir une colonne pour visualiser:",
            generated_data.columns,
            key="viz_col"
        )
        
        if generated_data[viz_col].dtype in ['object', 'category']:
            # Diagramme en barres pour les catégorielles
            fig = px.bar(
                generated_data[viz_col].value_counts().reset_index(),
                x='index',
                y=viz_col,
                title=f"Distribution de {viz_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Histogramme pour les numériques
            fig = px.histogram(
                generated_data,
                x=viz_col,
                title=f"Distribution de {viz_col}"
            )
            st.plotly_chart(fig, use_container_width=True)

def show_data_quality_report(generated_data):
    """
    Affiche un rapport de qualité des données
    """
    if generated_data is None:
        return
    
    st.markdown("### 🏆 Rapport de Qualité des Données")
    
    # Métriques de qualité
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        completeness = (1 - generated_data.isna().sum().sum() / (generated_data.shape[0] * generated_data.shape[1])) * 100
        st.metric("Complétude", f"{completeness:.1f}%")
    
    with col2:
        uniqueness = (generated_data.nunique() / len(generated_data)).mean() * 100
        st.metric("Diversité moyenne", f"{uniqueness:.1f}%")
    
    with col3:
        numeric_ratio = len(generated_data.select_dtypes(include=[np.number]).columns) / len(generated_data.columns) * 100
        st.metric("Variables numériques", f"{numeric_ratio:.1f}%")
    
    with col4:
        categorical_ratio = len(generated_data.select_dtypes(include=['object']).columns) / len(generated_data.columns) * 100
        st.metric("Variables catégorielles", f"{categorical_ratio:.1f}%")
    
    # Matrice des valeurs manquantes
    st.write("**Matrice des valeurs manquantes:**")
    missing_matrix = generated_data.isna()
    if missing_matrix.sum().sum() > 0:
        fig = px.imshow(
            missing_matrix,
            title="Valeurs manquantes (blanc = manquant)",
            color_continuous_scale=['green', 'white']
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ Aucune valeur manquante détectée!")