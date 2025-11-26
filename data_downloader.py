import pandas as pd
import streamlit as st
import io

# Gestion de l'import pyreadstat
try:
    from pyreadstat import write_sav
    PYREADSTAT_AVAILABLE = True
except ImportError:
    PYREADSTAT_AVAILABLE = False

class DataDownloader:
    """
    Classe pour gérer le téléchargement de données dans différents formats
    """
    
    def download_excel(self, df, filename):
        """
        Télécharge un DataFrame en format Excel
        """
        try:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Donnees', index=False)
                
                # Ajouter une feuille de métadonnées
                metadata = self._generate_metadata(df)
                metadata.to_excel(writer, sheet_name='Metadonnees', index=False)
            
            output.seek(0)
            
            st.download_button(
                label="📥 Cliquez pour télécharger le fichier Excel",
                data=output.getvalue(),
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"excel_{filename}"
            )
        except Exception as e:
            st.error(f"Erreur lors de l'export Excel: {str(e)}")
    
    def download_csv(self, df, filename):
        """
        Télécharge un DataFrame en format CSV
        """
        try:
            csv = df.to_csv(index=False, encoding='utf-8')
            
            st.download_button(
                label="📥 Cliquez pour télécharger le fichier CSV",
                data=csv,
                file_name=filename,
                mime="text/csv",
                key=f"csv_{filename}"
            )
        except Exception as e:
            st.error(f"Erreur lors de l'export CSV: {str(e)}")
    
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
                label="📥 Cliquez pour télécharger le fichier STATA",
                data=output.getvalue(),
                file_name=filename,
                mime="application/octet-stream",
                key=f"stata_{filename}"
            )
        except Exception as e:
            st.error(f"Erreur lors de l'export STATA: {str(e)}")
    
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
                label="📥 Cliquez pour télécharger le fichier SPSS",
                data=output.getvalue(),
                file_name=filename,
                mime="application/octet-stream",
                key=f"spss_{filename}"
            )
        except Exception as e:
            st.error(f"Erreur lors de l'export SPSS: {str(e)}")
    
    def download_txt(self, df, filename, delimiter='\t'):
        """
        Télécharge un DataFrame en format texte
        """
        try:
            txt_data = df.to_csv(sep=delimiter, index=False)
            
            st.download_button(
                label="📥 Cliquez pour télécharger le fichier TXT",
                data=txt_data,
                file_name=filename,
                mime="text/plain",
                key=f"txt_{filename}"
            )
        except Exception as e:
            st.error(f"Erreur lors de l'export TXT: {str(e)}")
    
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
                label="📊 Télécharger le rapport d'analyse",
                data=html_content,
                file_name=filename,
                mime="text/html",
                key="analysis_report"
            )
        except Exception as e:
            st.error(f"Erreur lors de la génération du rapport: {str(e)}")
    
    def _generate_html_report(self, analysis_results):
        """
        Génère un rapport HTML à partir des résultats d'analyse
        """
        import datetime
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rapport d'Analyse - LabMath Analytics</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .footer {{ background: #f8f9fa; padding: 10px; text-align: center; margin-top: 20px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔬 Rapport d'Analyse LabMath Analytics</h1>
                <p>Généré le {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
        """
        
        # Ajouter les sections d'analyse
        for section_name, section_data in analysis_results.items():
            html_content += f"""
            <div class="section">
                <h2>{section_name}</h2>
                <pre>{str(section_data)}</pre>
            </div>
            """
        
        html_content += f"""
            <div class="footer">
                <p>🔬 Powered by <strong>Lab_Math SCSM</strong> and <strong>CIE</strong> | © Copyright 2025</p>
                <p>Rapport généré automatiquement par LabMath Analytics Pro</p>
            </div>
        </body>
        </html>
        """
        
        return html_content