import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, ttest_ind, pearsonr, shapiro
import matplotlib.pyplot as plt
import seaborn as sns
from data_generator import DataGenerator

class AnalysisFunctions:
    def __init__(self):
        self.data_gen = DataGenerator()
    
    def generate_frequency_table(self, df, variable, group_variable, max_categories=15, mode="colonne"):
        """
        Génère un tableau de fréquences étendu selon le format demandé
        Utilise la nouvelle fonction generer_tableau_contingence_etendu
        """
        # Vérification des colonnes
        if variable not in df.columns or group_variable not in df.columns:
            raise ValueError(f"Variables non trouvées: {variable} ou {group_variable}")

        # Gérer les variables avec trop de catégories
        df_temp = self._gerer_categories_nombreuses(df, variable, max_categories)
        
        # Générer le tableau étendu
        tableau_etendu = self.data_gen.generer_tableau_contingence_etendu(
            df_temp, variable, group_variable, mode
        )
        
        return tableau_etendu

    def _gerer_categories_nombreuses(self, df, variable, max_categories):
        """Gère les variables avec trop de catégories"""
        if df[variable].nunique() > max_categories:
            value_counts = df[variable].value_counts()
            top_categories = value_counts.head(max_categories - 1).index
            df_temp = df.copy()
            df_temp[variable] = df_temp[variable].apply(
                lambda x: x if x in top_categories else 'Autres'
            )
            return df_temp
        return df.copy()

    def generate_frequency_table_classique(self, df, variable, group_variable, max_categories=15, mode="total"):
        """
        Version classique maintenue pour la compatibilité
        """
        # Vérification des colonnes
        if variable not in df.columns or group_variable not in df.columns:
            raise ValueError(f"Variables non trouvées: {variable} ou {group_variable}")

        # Gérer les variables avec trop de catégories
        if df[variable].nunique() > max_categories:
            value_counts = df[variable].value_counts()
            top_categories = value_counts.head(max_categories - 1).index
            df_temp = df.copy()
            df_temp[variable] = df_temp[variable].apply(
                lambda x: x if x in top_categories else 'Autres'
            )
        else:
            df_temp = df

        # Utiliser la fonction classique pour la cohérence
        try:
            return self.data_gen.generer_tableau_contingence_corrige(df_temp, variable, group_variable, mode)
        except:
            # Fallback si la fonction n'est pas disponible
            return self._generate_frequency_table_fallback(df_temp, variable, group_variable, mode)

    def _generate_frequency_table_fallback(self, df, var_ligne, var_col, mode="total"):
        """
        Version de fallback pour generer_tableau_contingence_corrige
        """
        # Tableau d'effectifs avec marges
        effectifs = pd.crosstab(
            df[var_ligne], 
            df[var_col], 
            margins=True, 
            margins_name="Total"
        )
        
        n_total = effectifs.loc["Total", "Total"]
        pourcent = effectifs.copy().astype(float)

        for i in effectifs.index:
            for j in effectifs.columns:
                nij = effectifs.loc[i, j]
                
                if mode == "total":
                    pourcent.loc[i, j] = 100 * nij / n_total if n_total > 0 else 0.0
                elif mode == "ligne":
                    if i == "Total":
                        pourcent.loc[i, j] = 100 * nij / n_total if n_total > 0 else 0.0
                    else:
                        denom = effectifs.loc[i, "Total"]
                        pourcent.loc[i, j] = 100 * nij / denom if denom > 0 else 0.0
                elif mode == "colonne":
                    if j == "Total":
                        pourcent.loc[i, j] = 100 * nij / n_total if n_total > 0 else 0.0
                    else:
                        denom = effectifs.loc["Total", j]
                        pourcent.loc[i, j] = 100 * nij / denom if denom > 0 else 0.0

        # Construction du tableau final
        final = effectifs.copy().astype(object)
        
        for i in effectifs.index:
            for j in effectifs.columns:
                e = effectifs.loc[i, j]
                p = round(float(pourcent.loc[i, j]), 1)

                if i != "Total" and j != "Total":
                    final.loc[i, j] = f"{e} ({p}%)"
                else:
                    final.loc[i, j] = f"{e}"

        return final

    def afficher_tableau_contingence(self, tableau):
        """
        Affiche un tableau de contingence formaté
        """
        return self.data_gen.formater_tableau_affichage(tableau)

    def test_chi2_carre(self, df, var1, var2):
        """
        Test du Chi-carré avec gestion d'erreurs améliorée
        """
        try:
            # Vérification des données
            if var1 not in df.columns or var2 not in df.columns:
                return {"error": f"Variables non trouvées: {var1} ou {var2}"}
            
            # Nettoyage des données
            clean_df = df[[var1, var2]].dropna()
            if len(clean_df) < 2:
                return {"error": "Données insuffisantes après nettoyage"}
            
            contingency_table = pd.crosstab(clean_df[var1], clean_df[var2])
            
            # Vérification de la taille du tableau
            if contingency_table.size < 4:
                return {"error": "Tableau de contingence trop petit"}
            
            # Vérification des effectifs minimums
            if (contingency_table < 5).sum().sum() > contingency_table.size * 0.2:
                warning = "Plus de 20% des cellules ont un effectif < 5, résultats potentiellement peu fiables"
            else:
                warning = None
                
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            
            # Calcul du V de Cramer
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            
            # Interprétation
            interpretation = self._interpret_chi2_result(p, cramers_v)
            
            result = {
                'test': 'Chi-carré',
                'chi2_statistic': round(chi2, 4),
                'p_value': round(p, 4),
                'degrees_freedom': dof,
                'cramers_v': round(cramers_v, 4),
                'contingency_table': contingency_table,
                'interpretation': interpretation,
                'warning': warning
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Erreur dans le test Chi-carré: {str(e)}"}

    def _interpret_chi2_result(self, p_value, cramers_v):
        """Interprète les résultats du test Chi-carré"""
        significance = "significatif" if p_value < 0.05 else "non significatif"
        
        if cramers_v < 0.1:
            effect_size = "faible"
        elif cramers_v < 0.3:
            effect_size = "moyen"
        else:
            effect_size = "fort"
            
        return f"Association {significance} (taille d'effet {effect_size})"

    def test_anova_ttest(self, df, cat_var, num_var):
        """
        ANOVA ou Test-t avec vérifications améliorées
        """
        try:
            # Vérifications
            if cat_var not in df.columns or num_var not in df.columns:
                return {"error": f"Variables non trouvées: {cat_var} ou {num_var}"}
            
            clean_df = df[[cat_var, num_var]].dropna()
            if len(clean_df) < 2:
                return {"error": "Données insuffisantes après nettoyage"}
            
            groups = clean_df.groupby(cat_var)[num_var]
            group_counts = groups.count()
            
            # Vérifier qu'il y a au moins 2 groupes avec des données
            valid_groups = group_counts[group_counts >= 2]
            if len(valid_groups) < 2:
                return {"error": "Nombre insuffisant de groupes avec données"}
            
            # Préparer les groupes pour le test
            group_data = [group for name, group in groups if len(group) >= 2]
            
            if len(group_data) == 2:
                # Test-t pour 2 groupes
                t_stat, p_value = ttest_ind(group_data[0], group_data[1], equal_var=False)
                test_type = 'Test-t'
                
                # Calcul de la taille d'effet (Cohen's d)
                mean1, mean2 = np.mean(group_data[0]), np.mean(group_data[1])
                std1, std2 = np.std(group_data[0], ddof=1), np.std(group_data[1], ddof=1)
                n1, n2 = len(group_data[0]), len(group_data[1])
                pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))
                cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                
                effect_size = cohens_d
                effect_size_name = "cohens_d"
                
            else:
                # ANOVA pour plus de 2 groupes
                f_stat, p_value = f_oneway(*group_data)
                test_type = 'ANOVA'
                
                # Calcul de eta carré
                ss_between = sum(len(group) * (np.mean(group) - np.mean(np.concatenate(group_data)))**2 
                               for group in group_data)
                ss_total = sum(np.sum((group - np.mean(np.concatenate(group_data)))**2) 
                             for group in group_data)
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                
                effect_size = eta_squared
                effect_size_name = "eta_squared"
            
            return {
                'test': test_type,
                'statistic': round(f_stat if test_type == 'ANOVA' else t_stat, 4),
                'p_value': round(p_value, 4),
                'effect_size': round(effect_size, 4),
                'effect_size_name': effect_size_name,
                'group_counts': group_counts.to_dict()
            }
            
        except Exception as e:
            return {"error": f"Erreur dans le test: {str(e)}"}

    def test_correlation(self, df, var1, var2):
        """
        Test de corrélation avec vérifications
        """
        try:
            if var1 not in df.columns or var2 not in df.columns:
                return {"error": f"Variables non trouvées: {var1} ou {var2}"}
            
            clean_data = df[[var1, var2]].dropna()
            if len(clean_data) < 3:
                return {"error": "Données insuffisantes pour le test de corrélation"}
            
            # Vérifier que les variables sont numériques
            if not (np.issubdtype(clean_data[var1].dtype, np.number) and 
                    np.issubdtype(clean_data[var2].dtype, np.number)):
                return {"error": "Les deux variables doivent être numériques"}
            
            corr_coef, p_value = pearsonr(clean_data[var1], clean_data[var2])
            
            # Interprétation
            if abs(corr_coef) < 0.3:
                strength = "faible"
            elif abs(corr_coef) < 0.7:
                strength = "modérée"
            else:
                strength = "forte"
                
            direction = "positive" if corr_coef > 0 else "négative"
            
            return {
                'test': 'Corrélation de Pearson',
                'correlation_coefficient': round(corr_coef, 4),
                'p_value': round(p_value, 4),
                'interpretation': f"Corrélation {strength} et {direction}",
                'sample_size': len(clean_data)
            }
            
        except Exception as e:
            return {"error": f"Erreur dans le test de corrélation: {str(e)}"}

    def create_visualization(self, df, viz_type, **kwargs):
        """
        Fonction unifiée pour créer des visualisations
        """
        try:
            if viz_type == 'bar':
                return self.create_bar_chart(df, **kwargs)
            elif viz_type == 'histogram':
                return self.create_histogram(df, **kwargs)
            elif viz_type == 'boxplot':
                return self.create_boxplot(df, **kwargs)
            elif viz_type == 'scatter':
                return self.create_scatter_plot(df, **kwargs)
            elif viz_type == 'correlation':
                return self.create_correlation_matrix(df, **kwargs)
            else:
                raise ValueError(f"Type de visualisation non supporté: {viz_type}")
                
        except Exception as e:
            print(f"Erreur dans la création de la visualisation: {str(e)}")
            return None

    def create_bar_chart(self, df, x_var, color_var=None, title=None):
        """Crée un diagramme en barres"""
        if title is None:
            title = f"Distribution de {x_var}" + (f" par {color_var}" if color_var else "")
        
        fig = px.histogram(df, x=x_var, color=color_var, barmode='group', title=title)
        return fig

    def create_histogram(self, df, num_var, color_var=None, title=None):
        """Crée un histogramme"""
        if title is None:
            title = f"Distribution de {num_var}"
        
        fig = px.histogram(df, x=num_var, color=color_var, marginal="box", title=title)
        return fig

    def create_boxplot(self, df, cat_var, num_var, title=None):
        """Crée un boxplot"""
        if title is None:
            title = f"Distribution de {num_var} par {cat_var}"
        
        fig = px.box(df, x=cat_var, y=num_var, title=title)
        return fig

    def create_scatter_plot(self, df, x_var, y_var, color_var=None, title=None):
        """Crée un scatter plot"""
        if title is None:
            title = f"Relation entre {x_var} et {y_var}"
        
        fig = px.scatter(df, x=x_var, y=y_var, color=color_var, title=title)
        return fig

    def create_correlation_matrix(self, df, numerical_vars=None):
        """Crée une matrice de corrélation"""
        if numerical_vars is None:
            numerical_vars = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_vars) < 2:
            return None
        
        corr_matrix = df[numerical_vars].corr()
        fig = px.imshow(corr_matrix, title="Matrice de Corrélation", aspect="auto")
        return fig, corr_matrix

    def calculate_descriptive_stats(self, df, variables=None):
        """
        Calcule les statistiques descriptives avec plus d'indicateurs
        """
        if variables is None:
            variables = df.select_dtypes(include=[np.number, 'category', 'object']).columns
        
        stats_dict = {}
        
        for var in variables:
            if np.issubdtype(df[var].dtype, np.number):
                # Statistiques pour variables numériques
                clean_data = df[var].dropna()
                stats_dict[var] = {
                    'type': 'Numérique',
                    'count': len(clean_data),
                    'missing': df[var].isna().sum(),
                    'missing_percent': round(100 * df[var].isna().sum() / len(df), 1),
                    'mean': round(clean_data.mean(), 2),
                    'std': round(clean_data.std(), 2),
                    'min': round(clean_data.min(), 2),
                    '25%': round(clean_data.quantile(0.25), 2),
                    'median': round(clean_data.median(), 2),
                    '75%': round(clean_data.quantile(0.75), 2),
                    'max': round(clean_data.max(), 2),
                    'skewness': round(clean_data.skew(), 2),
                    'kurtosis': round(clean_data.kurtosis(), 2)
                }
            else:
                # Statistiques pour variables catégorielles
                value_counts = df[var].value_counts()
                stats_dict[var] = {
                    'type': 'Catégorielle',
                    'count': df[var].count(),
                    'missing': df[var].isna().sum(),
                    'missing_percent': round(100 * df[var].isna().sum() / len(df), 1),
                    'unique': df[var].nunique(),
                    'mode': value_counts.index[0] if len(value_counts) > 0 else None,
                    'freq_mode': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    'freq_mode_percent': round(100 * value_counts.iloc[0] / len(df[var].dropna()), 1) if len(value_counts) > 0 else 0,
                    'categories': value_counts.to_dict()
                }
        
        return stats_dict

    def perform_comprehensive_analysis(self, df, target_var=None):
        """
        Effectue une analyse complète du dataset
        """
        analysis_results = {}
        
        # Statistiques descriptives
        analysis_results['descriptive_stats'] = self.calculate_descriptive_stats(df)
        
        # Matrice de corrélation (si variables numériques)
        numerical_vars = df.select_dtypes(include=[np.number]).columns
        if len(numerical_vars) >= 2:
            corr_result = self.create_correlation_matrix(df, numerical_vars)
            if corr_result:
                analysis_results['correlation_matrix'] = corr_result
        
        return analysis_results

    def analyze_target_variable(self, df, target_var):
        """
        Analyse approfondie d'une variable cible
        """
        analysis = {}
        
        if np.issubdtype(df[target_var].dtype, np.number):
            # Variable numérique
            analysis['type'] = 'numerical'
            analysis['distribution'] = {
                'mean': df[target_var].mean(),
                'median': df[target_var].median(),
                'std': df[target_var].std(),
                'skewness': df[target_var].skew()
            }
        else:
            # Variable catégorielle
            analysis['type'] = 'categorical'
            value_counts = df[target_var].value_counts()
            analysis['distribution'] = {
                'categories': value_counts.to_dict(),
                'proportions': (value_counts / len(df)).to_dict()
            }
        
        return analysis

# Fonctions utilitaires pour l'export
def format_statistical_result(result):
    """Formate les résultats statistiques pour l'affichage"""
    if 'error' in result:
        return f"❌ Erreur: {result['error']}"
    
    formatted = []
    for key, value in result.items():
        if key not in ['contingency_table', 'interpretation', 'warning']:
            if isinstance(value, float):
                formatted.append(f"{key}: {value:.4f}")
            else:
                formatted.append(f"{key}: {value}")
    
    if 'interpretation' in result:
        formatted.append(f"📊 Interprétation: {result['interpretation']}")
    
    if 'warning' in result and result['warning']:
        formatted.append(f"⚠️ Attention: {result['warning']}")
    
    return "\n".join(formatted)

# Instance globale pour une utilisation facile
analysis = AnalysisFunctions()