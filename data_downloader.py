import pandas as pd
import streamlit as st
import io
from pyreadstat import write_sav

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
                data=output,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
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
                mime="text/csv"
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
                mime="application/octet-stream"
            )
        except Exception as e:
            st.error(f"Erreur lors de l'export STATA: {str(e)}")
    
    def download_spss(self, df, filename):
        """
        Télécharge un DataFrame en format SPSS
        """
        try:
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
                mime="application/octet-stream"
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
                mime="text/plain"
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