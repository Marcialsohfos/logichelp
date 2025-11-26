import pandas as pd
import streamlit as st
import io
import numpy as np
from typing import Optional, List

# Gestion de l'import pyreadstat
try:
    from pyreadstat import write_sav
    PYREADSTAT_AVAILABLE = True
except ImportError:
    PYREADSTAT_AVAILABLE = False

class DataDownloader:
    """
    Classe pour gérer le téléchargement ET l'upload de données dans différents formats
    """
    
    def upload_data(self, allowed_types: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """
        Upload un fichier de données et retourne un DataFrame
        
        Parameters:
        allowed_types: Liste des types de fichiers autorisés
            
        Returns:
        DataFrame ou None si échec
        """
        if allowed_types is None:
            allowed_types = ['xlsx', 'xls', 'csv', 'dta', 'txt']
        
        uploaded_file = st.file_uploader(
            "📤 Choisissez votre fichier de données",
            type=allowed_types,
            help=f"Formats supportés: {', '.join(allowed_types).upper()}"
        )
        
        if uploaded_file is not None:
            try:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                df = self._read_uploaded_file(uploaded_file, file_extension)
                
                if df is not None and not df.empty:
                    st.success(f"✅ Fichier '{uploaded_file.name}' chargé avec succès! ({df.shape[0]} lignes, {df.shape[1]} colonnes)")
                    return df
                else:
                    st.error("❌ Le fichier est vide ou n'a pas pu être lu correctement")
                    
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement du fichier: {str(e)}")
        
        return None
    
    def _read_uploaded_file(self, uploaded_file, file_extension: str) -> Optional[pd.DataFrame]:
        """
        Lit un fichier uploadé selon son format
        """
        try:
            if file_extension in ['xlsx', 'xls']:
                return self._read_excel(uploaded_file)
            elif file_extension == 'csv':
                return pd.read_csv(uploaded_file)
            elif file_extension == 'dta':
                return pd.read_stata(uploaded_file)
            elif file_extension == 'sav':
                return self._read_spss(uploaded_file)
            elif file_extension == 'txt':
                return self._read_text(uploaded_file)
            else:
                st.error(f"❌ Format {file_extension} non supporté")
                return None
                
        except Exception as e:
            st.error(f"❌ Erreur lors de la lecture du fichier {file_extension}: {str(e)}")
            return None
    
    def _read_excel(self, uploaded_file) -> pd.DataFrame:
        """
        Lit un fichier Excel avec gestion des onglets multiples
        """
        try:
            # Lire le fichier Excel
            excel_file = pd.ExcelFile(uploaded_file)
            
            if len(excel_file.sheet_names) > 1:
                # Si plusieurs onglets, laisser l'utilisateur choisir
                sheet_name = st.selectbox(
                    "📋 Choisissez l'onglet à charger:",
                    options=excel_file.sheet_names,
                    index=0
                )
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
            else:
                # Un seul onglet
                df = pd.read_excel(uploaded_file)
            
            # Nettoyer les noms de colonnes
            df.columns = [str(col).strip() for col in df.columns]
            
            return df
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la lecture du fichier Excel: {str(e)}")
            return pd.DataFrame()
    
    def _read_spss(self, uploaded_file) -> Optional[pd.DataFrame]:
        """
        Lit un fichier SPSS (.sav)
        """
        if not PYREADSTAT_AVAILABLE:
            st.error("❌ Le format SPSS (.sav) n'est pas supporté sur cette plateforme")
            st.info("💡 Utilisez Excel, CSV ou STATA à la place")
            return None
        
        try:
            from pyreadstat import read_sav
            df, meta = read_sav(uploaded_file)
            return df
        except Exception as e:
            st.error(f"❌ Erreur lors de la lecture du fichier SPSS: {str(e)}")
            return None
    
    def _read_text(self, uploaded_file) -> pd.DataFrame:
        """
        Lit un fichier texte avec détection automatique du séparateur
        """
        try:
            # Lire les premières lignes pour détecter le séparateur
            content = uploaded_file.getvalue().decode('utf-8')
            first_lines = content.split('\n')[:5]
            
            # Détecter le séparateur le plus probable
            separators = [',', ';', '\t', '|']
            best_separator = ','
            max_count = 0
            
            for sep in separators:
                if first_lines:
                    count = first_lines[0].count(sep)
                    if count > max_count:
                        max_count = count
                        best_separator = sep
            
            st.info(f"🔍 Séparateur détecté: '{best_separator}'")
            
            # Réessayer la lecture
            uploaded_file.seek(0)  # Reset file pointer
            return pd.read_csv(uploaded_file, sep=best_separator, encoding='utf-8')
            
        except UnicodeDecodeError:
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, sep=best_separator, encoding='latin-1')
            except Exception as e:
                st.error(f"❌ Impossible de lire le fichier texte: {str(e)}")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"❌ Erreur lors de la lecture du fichier texte: {str(e)}")
            return pd.DataFrame()
    
    def show_file_preview(self, df: pd.DataFrame, max_rows: int = 10):
        """
        Affiche un aperçu du fichier chargé
        """
        if df is None or df.empty:
            st.warning("⚠️ Aucune donnée à afficher")
            return
        
        st.markdown("### 👀 Aperçu des données")
        
        # Onglets pour différentes vues
        tab1, tab2, tab3 = st.tabs(["📋 Données", "📊 Résumé", "🔍 Types"])
        
        with tab1:
            st.dataframe(df.head(max_rows), use_container_width=True)
            st.write(f"**Dimensions:** {df.shape[0]} lignes × {df.shape[1]} colonnes")
            
            # Informations basiques
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Lignes", df.shape[0])
            with col2:
                st.metric("Colonnes", df.shape[1])
            with col3:
                st.metric("Valeurs manquantes", df.isna().sum().sum())
        
        with tab2:
            st.write("**Statistiques descriptives (variables numériques):**")
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                st.dataframe(df[numerical_cols].describe())
            else:
                st.info("Aucune variable numérique trouvée")
            
            st.write("**Variables catégorielles:**")
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols[:5]:  # Limiter à 5 colonnes
                st.write(f"**{col}:** {df[col].nunique()} catégories uniques")
        
        with tab3:
            st.write("**Types de données:**")
            type_summary = df.dtypes.value_counts()
            for dtype, count in type_summary.items():
                st.write(f"- **{dtype}:** {count} colonne(s)")
            
            st.write("**Liste des colonnes:**")
            for col in df.columns:
                dtype = df[col].dtype
                unique_vals = df[col].nunique()
                missing_vals = df[col].isna().sum()
                st.write(f"- **{col}** ({dtype}) - {unique_vals} uniques - {missing_vals} manquants")

    def download_excel(self, df, filename):
        """
        Télécharge un DataFrame en format Excel
        """
        try:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Feuille principale avec les données
                df.to_excel(writer, sheet_name='Donnees', index=False)
                
                # Feuille de métadonnées
                metadata = self._generate_metadata(df)
                metadata.to_excel(writer, sheet_name='Metadonnees', index=False)
                
                # Feuille de statistiques descriptives
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                if len(numerical_cols) > 0:
                    df[numerical_cols].describe().to_excel(writer, sheet_name='Statistiques')
            
            output.seek(0)
            
            st.download_button(
                label="📥 Télécharger le fichier Excel",
                data=output.getvalue(),
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"excel_{filename}_{pd.Timestamp.now().value}"
            )
        except Exception as e:
            st.error(f"❌ Erreur lors de l'export Excel: {str(e)}")
    
    def download_csv(self, df, filename):
        """
        Télécharge un DataFrame en format CSV
        """
        try:
            csv = df.to_csv(index=False, encoding='utf-8')
            
            st.download_button(
                label="📥 Télécharger le fichier CSV",
                data=csv,
                file_name=filename,
                mime="text/csv",
                key=f"csv_{filename}_{pd.Timestamp.now().value}"
            )
        except Exception as e:
            st.error(f"❌ Erreur lors de l'export CSV: {str(e)}")
    
    def download_stata(self, df, filename):
        """
        Télécharge un DataFrame en format STATA
        """
        try:
            output = io.BytesIO()
            
            # Convertir les colonnes object en string pour STATA
            df_stata = df.copy()
            for col in df_stata.select_dtypes(include=['object']).columns:
                df_stata[col] = df_stata[col].astype(str)
            
            df_stata.to_stata(output, write_index=False)
            output.seek(0)
            
            st.download_button(
                label="📥 Télécharger le fichier STATA",
                data=output.getvalue(),
                file_name=filename,
                mime="application/octet-stream",
                key=f"stata_{filename}_{pd.Timestamp.now().value}"
            )
        except Exception as e:
            st.error(f"❌ Erreur lors de l'export STATA: {str(e)}")
    
    def download_spss(self, df, filename):
        """
        Télécharge un DataFrame en format SPSS
        """
        try:
            if not PYREADSTAT_AVAILABLE:
                st.error("❌ L'export SPSS n'est pas disponible sur cette plateforme")
                st.info("💡 Utilisez Excel ou CSV à la place")
                return
            
            output = io.BytesIO()
            
            # Préparer les données pour SPSS
            df_spss = df.copy()
            variable_labels = {col: col for col in df_spss.columns}
            
            write_sav(df_spss, output, variable_labels=variable_labels)
            output.seek(0)
            
            st.download_button(
                label="📥 Télécharger le fichier SPSS",
                data=output.getvalue(),
                file_name=filename,
                mime="application/octet-stream",
                key=f"spss_{filename}_{pd.Timestamp.now().value}"
            )
        except Exception as e:
            st.error(f"❌ Erreur lors de l'export SPSS: {str(e)}")
    
    def download_txt(self, df, filename, delimiter='\t'):
        """
        Télécharge un DataFrame en format texte
        """
        try:
            txt_data = df.to_csv(sep=delimiter, index=False)
            
            st.download_button(
                label="📥 Télécharger le fichier TXT",
                data=txt_data,
                file_name=filename,
                mime="text/plain",
                key=f"txt_{filename}_{pd.Timestamp.now().value}"
            )
        except Exception as e:
            st.error(f"❌ Erreur lors de l'export TXT: {str(e)}")
    
    def _generate_metadata(self, df):
        """
        Génère un DataFrame de métadonnées
        """
        metadata = {
            'Variable': df.columns.tolist(),
            'Type': [str(dtype) for dtype in df.dtypes],
            'Valeurs Manquantes': [df[col].isna().sum() for col in df.columns],
            'Valeurs Uniques': [df[col].nunique() for col in df.columns],
            'Exemple Valeurs': [self._get_example_values(df[col]) for col in df.columns]
        }
        
        return pd.DataFrame(metadata)
    
    def _get_example_values(self, series):
        """
        Retourne des exemples de valeurs pour une série
        """
        non_null_values = series.dropna()
        if len(non_null_values) > 0:
            examples = non_null_values.head(3).tolist()
            return ', '.join(map(str, examples))
        return 'Aucune valeur'
    
    def download_analysis_report(self, analysis_results, filename="rapport_analyse.html"):
        """
        Télécharge un rapport d'analyse complet
        """
        try:
            html_content = self._generate_html_report(analysis_results)
            
            st.download_button(
                label="📊 Télécharger le rapport d'analyse complet",
                data=html_content,
                file_name=filename,
                mime="text/html",
                key=f"report_{pd.Timestamp.now().value}"
            )
        except Exception as e:
            st.error(f"❌ Erreur lors de la génération du rapport: {str(e)}")
    
    def _generate_html_report(self, analysis_results):
        """
        Génère un rapport HTML à partir des résultats d'analyse
        """
        import datetime
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rapport d'Analyse - LogicApp Analytics</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }}
                .footer {{ background: #f8f9fa; padding: 15px; text-align: center; margin-top: 30px; border-top: 2px solid #667eea; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
                th {{ background-color: #667eea; color: white; }}
                .metric {{ background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔬 Rapport d'Analyse LogicApp Analytics</h1>
                <p>Généré le {datetime.datetime.now().strftime("%Y-%m-%d à %H:%M:%S")}</p>
            </div>
        """
        
        # Ajouter les sections d'analyse
        for section_name, section_data in analysis_results.items():
            html_content += f"""
            <div class="section">
                <h2>📈 {section_name}</h2>
                <div class="metric">
                    <pre>{str(section_data)}</pre>
                </div>
            </div>
            """
        
        html_content += f"""
            <div class="footer">
                <p><strong>🔬 Powered by LogicApp Analytics Pro</strong></p>
                <p>© Copyright 2025 - Tous droits réservés</p>
                <p>Rapport généré automatiquement par LogicApp Analytics Pro</p>
            </div>
        </body>
        </html>
        """
        
        return html_content