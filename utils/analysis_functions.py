import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, ttest_ind, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

def generate_conditional_table(df, row_variable, column_variable, percentage_type='col'):
    """
    Génère un tableau conditionnel formaté avec effectifs et pourcentages CORRECTS
    """
    # Créer le tableau croisé de base sans marges d'abord
    cross_tab = pd.crosstab(
        df[row_variable], 
        df[column_variable],
        margins=False
    )
    
    # Calculer les totaux manuellement pour garantir l'exactitude
    row_totals = cross_tab.sum(axis=1)
    col_totals = cross_tab.sum(axis=0)
    grand_total = cross_tab.sum().sum()
    
    # Ajouter les totaux manuellement
    cross_tab_with_margins = cross_tab.copy()
    cross_tab_with_margins['Total'] = row_totals
    total_row = col_totals.copy()
    total_row['Total'] = grand_total
    cross_tab_with_margins.loc['Total'] = total_row
    
    # Calculer les pourcentages CORRECTS
    if percentage_type == 'col':
        # Pourcentages par colonne
        percent_data = []
        for row_idx, row_name in enumerate(cross_tab_with_margins.index):
            row_percents = []
            for col_idx, col_name in enumerate(cross_tab_with_margins.columns):
                count = cross_tab_with_margins.iloc[row_idx, col_idx]
                if col_name == 'Total':
                    # Pour la colonne Total, utiliser le total général comme dénominateur
                    percent = (count / grand_total * 100) if grand_total > 0 else 0
                else:
                    # Pour les autres colonnes, utiliser le total de la colonne
                    col_total = cross_tab_with_margins.loc['Total', col_name]
                    percent = (count / col_total * 100) if col_total > 0 else 0
                row_percents.append(percent)
            percent_data.append(row_percents)
        
        percent_df = pd.DataFrame(percent_data, 
                                index=cross_tab_with_margins.index,
                                columns=cross_tab_with_margins.columns)
    
    elif percentage_type == 'row':
        # Pourcentages par ligne
        percent_data = []
        for row_idx, row_name in enumerate(cross_tab_with_margins.index):
            row_percents = []
            for col_idx, col_name in enumerate(cross_tab_with_margins.columns):
                count = cross_tab_with_margins.iloc[row_idx, col_idx]
                if row_name == 'Total':
                    # Pour la ligne Total, utiliser le total général comme dénominateur
                    percent = (count / grand_total * 100) if grand_total > 0 else 0
                else:
                    # Pour les autres lignes, utiliser le total de la ligne
                    row_total = cross_tab_with_margins.loc[row_name, 'Total']
                    percent = (count / row_total * 100) if row_total > 0 else 0
                row_percents.append(percent)
            percent_data.append(row_percents)
        
        percent_df = pd.DataFrame(percent_data,
                                index=cross_tab_with_margins.index,
                                columns=cross_tab_with_margins.columns)
    
    else:  # 'all' - pourcentages du total général
        percent_df = cross_tab_with_margins / grand_total * 100
    
    # Arrondir les pourcentages
    percent_df = percent_df.round(1)
    
    # Créer le tableau combiné final
    combined_data = []
    for row_idx, row_name in enumerate(cross_tab_with_margins.index):
        row_data = []
        for col_idx, col_name in enumerate(cross_tab_with_margins.columns):
            count = cross_tab_with_margins.iloc[row_idx, col_idx]
            percent = percent_df.iloc[row_idx, col_idx]
            row_data.extend([count, percent])
        combined_data.append(row_data)
    
    # Créer les noms de colonnes
    column_names = []
    for col_name in cross_tab_with_margins.columns:
        column_names.extend([col_name, '%'])
    
    # Créer le DataFrame final
    result_df = pd.DataFrame(
        combined_data,
        index=cross_tab_with_margins.index,
        columns=column_names
    )
    
    return result_df

def format_conditional_table(conditional_df, title="Tableau Conditionnel"):
    """
    Formate le tableau conditionnel avec un style professionnel
    """
    # Formater les nombres
    styled_df = conditional_df.copy()
    for col in styled_df.columns:
        if col == '%' or '%' in str(col):
            # Formater les pourcentages avec virgule
            styled_df[col] = styled_df[col].apply(
                lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x
            )
        else:
            # Formater les entiers
            styled_df[col] = styled_df[col].apply(
                lambda x: f"{int(x)}" if isinstance(x, (int, float)) and x == int(x) else x
            )
    
    # Appliquer le style
    styled_table = styled_df.style.set_properties(**{
        'text-align': 'center',
        'border': '1px solid black',
        'padding': '8px',
        'font-size': '12px',
        'font-family': 'Arial, sans-serif'
    }).set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#366092'), 
                                   ('color', 'white'), 
                                   ('font-weight', 'bold'),
                                   ('text-align', 'center'),
                                   ('border', '1px solid black')]},
        {'selector': 'td', 'props': [('border', '1px solid black'),
                                   ('padding', '8px')]},
        {'selector': 'thead th', 'props': [('background-color', '#2F5597'), 
                                         ('color', 'white'), 
                                         ('font-weight', 'bold')]},
        {'selector': '.index_name', 'props': [('background-color', '#366092'), 
                                            ('color', 'white'), 
                                            ('font-weight', 'bold')]},
        {'selector': '.row_heading', 'props': [('background-color', '#D9E2F3'),
                                             ('font-weight', 'bold')]}
    ]).set_caption(
        f"<b>{title}</b><br>"
        f"<span style='font-size: 12px;'>Effectifs et pourcentages (%)</span>"
    )
    
    return styled_table

def create_healthcare_analysis(df):
    """
    Crée une analyse complète pour le domaine de la santé
    """
    analyses = {}
    
    # Tableau conditionnel principal : Type d'établissement par Niveau de complexité
    if 'Type_Etablissement' in df.columns and 'Niveau_Complexite' in df.columns:
        main_table = generate_conditional_table(df, 'Type_Etablissement', 'Niveau_Complexite', 'col')
        analyses['type_etablissement_niveau'] = format_conditional_table(
            main_table, 
            "Répartition des établissements par type et niveau de complexité"
        )
    
    # Tableau conditionnel : Glycosurie/Albuminurie par Niveau de complexité
    if 'Glycosurie_Albuminurie' in df.columns and 'Niveau_Complexite' in df.columns:
        medical_table = generate_conditional_table(df, 'Glycosurie_Albuminurie', 'Niveau_Complexite', 'all')
        analyses['medical_niveau'] = format_conditional_table(
            medical_table,
            "Prévalence de la glycosurie et albuminurie par niveau de complexité"
        )
    
    return analyses

# Fonctions statistiques conservées (sans les tableaux de contingence problématiques)
def test_chi2_carre(df, var1, var2):
    """
    Test du Chi-carré pour deux variables catégorielles
    """
    contingency_table = pd.crosstab(df[var1], df[var2])
    
    if contingency_table.size == 0:
        return "Tableau de contingence vide"
    
    try:
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        
        result = {
            'test': 'Chi-carré',
            'chi2_statistic': chi2,
            'p_value': p,
            'degrees_freedom': dof
        }
        
        return result
    except Exception as e:
        return f"Erreur dans le test Chi-carré: {str(e)}"

def test_anova_ttest(df, cat_var, num_var):
    """
    ANOVA ou Test-t pour variable catégorielle vs numérique
    """
    groups = df.groupby(cat_var)[num_var].apply(list)
    
    if len(groups) == 2:
        # Test-t pour 2 groupes
        t_stat, p_value = ttest_ind(groups.iloc[0], groups.iloc[1])
        return {
            'test': 'Test-t',
            't_statistic': t_stat,
            'p_value': p_value
        }
    else:
        # ANOVA pour plus de 2 groupes
        f_stat, p_value = f_oneway(*groups)
        return {
            'test': 'ANOVA',
            'f_statistic': f_stat,
            'p_value': p_value
        }

def test_correlation(df, var1, var2):
    """
    Test de corrélation pour deux variables numériques
    """
    corr_coef, p_value = pearsonr(df[var1].dropna(), df[var2].dropna())
    
    return {
        'test': 'Corrélation de Pearson',
        'correlation_coefficient': corr_coef,
        'p_value': p_value
    }

def create_bar_chart(df, x_var, color_var=None):
    """
    Crée un diagramme en barres interactif
    """
    if color_var:
        fig = px.histogram(df, x=x_var, color=color_var, barmode='group',
                          title=f"Distribution de {x_var} par {color_var}")
    else:
        fig = px.histogram(df, x=x_var, title=f"Distribution de {x_var}")
    
    fig.update_layout(
        xaxis_title=x_var,
        yaxis_title="Effectif",
        legend_title=color_var
    )
    
    return fig

def create_histogram(df, num_var, color_var=None):
    """
    Crée un histogramme
    """
    if color_var:
        fig = px.histogram(df, x=num_var, color=color_var, marginal="box",
                          title=f"Distribution de {num_var}")
    else:
        fig = px.histogram(df, x=num_var, title=f"Distribution de {num_var}")
    
    return fig

def create_boxplot(df, cat_var, num_var):
    """
    Crée un boxplot
    """
    fig = px.box(df, x=cat_var, y=num_var, 
                title=f"Distribution de {num_var} par {cat_var}")
    return fig

def calculate_descriptive_stats(df, variables=None):
    """
    Calcule les statistiques descriptives pour les variables spécifiées
    """
    if variables is None:
        variables = df.columns
    
    stats_dict = {}
    
    for var in variables:
        if df[var].dtype in ['int64', 'float64']:
            # Statistiques pour variables numériques
            stats_dict[var] = {
                'type': 'Numérique',
                'count': df[var].count(),
                'mean': df[var].mean(),
                'std': df[var].std(),
                'min': df[var].min(),
                '25%': df[var].quantile(0.25),
                'median': df[var].median(),
                '75%': df[var].quantile(0.75),
                'max': df[var].max(),
                'missing': df[var].isna().sum()
            }
        else:
            # Statistiques pour variables catégorielles
            stats_dict[var] = {
                'type': 'Catégorielle',
                'count': df[var].count(),
                'unique': df[var].nunique(),
                'mode': df[var].mode().iloc[0] if not df[var].empty else None,
                'freq_mode': df[var].value_counts().iloc[0] if not df[var].empty else 0,
                'missing': df[var].isna().sum()
            }
    
    return stats_dict

def create_correlation_matrix(df, numerical_vars=None):
    """
    Crée une matrice de corrélation pour les variables numériques
    """
    if numerical_vars is None:
        numerical_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_vars) < 2:
        return None
    
    corr_matrix = df[numerical_vars].corr()
    
    # Créer un heatmap interactif
    fig = px.imshow(corr_matrix,
                   title="Matrice de Corrélation",
                   color_continuous_scale='RdBu_r',
                   aspect="auto")
    
    return fig, corr_matrix

# Exemple d'utilisation
if __name__ == "__main__":
    # Créer un générateur de données
    generator = DataGenerator()
    
    # Générer un dataset santé
    df = generator.generate_healthcare_dataset(548)  # Même taille que votre exemple
    
    # Créer une analyse complète
    analyses = create_healthcare_analysis(df)
    
    # Afficher les tableaux
    for name, table in analyses.items():
        print(f"\n{name}:")
        display(table)