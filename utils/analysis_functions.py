import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, ttest_ind, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

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

def display_frequency_table(table_df, show_percentages=True, show_totals=True):
    """
    Affiche un tableau de fréquences formaté
    """
    # Appliquer le style
    styled_table = table_df.style.set_properties(**{
        'text-align': 'center',
        'border': '1px solid black'
    }).set_table_styles([{
        'selector': 'th',
        'props': [('background-color', '#366092'), ('color', 'white')]
    }])
    
    return styled_table

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
            'degrees_freedom': dof,
            'contingency_table': contingency_table
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

def create_bar_chart(df, x_var, color_var=None, group_var=None):
    """
    Crée un diagramme en barres interactif
    """
    if group_var:
        fig = px.histogram(df, x=x_var, color=color_var, barmode='group',
                          title=f"Distribution de {x_var} par {color_var}")
    else:
        fig = px.histogram(df, x=x_var, color=color_var,
                          title=f"Distribution de {x_var} par {color_var}")
    
    fig.update_layout(
        xaxis_title=x_var,
        yaxis_title="Effectif",
        legend_title=color_var
    )
    
    return fig

def create_stacked_bar_chart(df, x_var, stack_var=None, color_var=None):
    """
    Crée un diagramme en bande (stacked bar chart)
    """
    if color_var:
        fig = px.histogram(df, x=x_var, color=stack_var, barmode='stack',
                          facet_col=color_var, title=f"Diagramme en bande: {x_var} par {stack_var}")
    else:
        fig = px.histogram(df, x=x_var, color=stack_var, barmode='stack',
                          title=f"Diagramme en bande: {x_var} par {stack_var}")
    
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

def create_scatter_plot(df, x_var, y_var, color_var=None):
    """
    Crée un scatter plot
    """
    if color_var:
        fig = px.scatter(df, x=x_var, y=y_var, color=color_var,
                        title=f"Relation entre {x_var} et {y_var}")
    else:
        fig = px.scatter(df, x=x_var, y=y_var,
                        title=f"Relation entre {x_var} et {y_var}")
    
    # Ajouter la ligne de régression si les données sont numériques
    if df[x_var].dtype in ['int64', 'float64'] and df[y_var].dtype in ['int64', 'float64']:
        try:
            # Calculer la régression linéaire
            clean_data = df[[x_var, y_var]].dropna()
            if len(clean_data) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(clean_data[x_var], clean_data[y_var])
                line_x = np.array([clean_data[x_var].min(), clean_data[x_var].max()])
                line_y = intercept + slope * line_x
                
                fig.add_trace(go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode='lines',
                    name=f'Régression (r={r_value:.2f})',
                    line=dict(color='red', dash='dash')
                ))
        except:
            pass  # Ignorer les erreurs de régression
    
    return fig

def create_3d_visualization(df, var1, var2, var3, color_var):
    """
    Crée une visualisation 3D interactive
    """
    # Scatter plot 3D
    fig = px.scatter_3d(df, x=var1, y=var2, z=var3, color=color_var,
                       title=f"Visualisation 3D: {var1}, {var2}, {var3}")
    
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

def perform_normality_test(df, numerical_vars=None):
    """
    Effectue le test de normalité de Shapiro-Wilk sur les variables numériques
    """
    if numerical_vars is None:
        numerical_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    
    normality_results = {}
    
    for var in numerical_vars:
        clean_data = df[var].dropna()
        if len(clean_data) >= 3 and len(clean_data) <= 5000:  # Limitations du test Shapiro-Wilk
            stat, p_value = stats.shapiro(clean_data)
            normality_results[var] = {
                'statistic': stat,
                'p_value': p_value,
                'normal': p_value > 0.05
            }
        else:
            normality_results[var] = {
                'statistic': None,
                'p_value': None,
                'normal': None,
                'message': 'Échantillon trop petit ou trop grand pour Shapiro-Wilk'
            }
    
    return normality_results

def create_interactive_distribution(df, variable, group_var=None):
    """
    Crée une visualisation interactive de la distribution
    """
    if df[variable].dtype in ['int64', 'float64']:
        # Variable numérique - histogramme + boxplot
        if group_var:
            fig = px.histogram(df, x=variable, color=group_var, 
                             marginal="box", barmode="overlay",
                             title=f"Distribution de {variable} par {group_var}")
        else:
            fig = px.histogram(df, x=variable, marginal="box",
                             title=f"Distribution de {variable}")
    else:
        # Variable catégorielle - diagramme en barres
        if group_var:
            fig = px.histogram(df, x=variable, color=group_var, barmode='group',
                             title=f"Distribution de {variable} par {group_var}")
        else:
            value_counts = df[variable].value_counts().reset_index()
            value_counts.columns = [variable, 'count']
            fig = px.bar(value_counts, x=variable, y='count',
                        title=f"Distribution de {variable}")
    
    return fig

def calculate_effect_size(df, var1, var2):
    """
    Calcule la taille d'effet pour différentes combinaisons de variables
    """
    effect_sizes = {}
    
    # Vérifier les types de variables
    var1_type = 'catégorielle' if df[var1].dtype == 'object' or df[var1].nunique() < 10 else 'numérique'
    var2_type = 'catégorielle' if df[var2].dtype == 'object' or df[var2].nunique() < 10 else 'numérique'
    
    if var1_type == 'catégorielle' and var2_type == 'catégorielle':
        # V de Cramer pour deux variables catégorielles
        contingency_table = pd.crosstab(df[var1], df[var2])
        chi2, _, _, _ = chi2_contingency(contingency_table)
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        effect_sizes['cramers_v'] = cramers_v
        
    elif var1_type == 'catégorielle' and var2_type == 'numérique':
        # Eta carré pour variable catégorielle vs numérique
        groups = [group for name, group in df.groupby(var1)[var2]]
        f_stat, _ = f_oneway(*groups)
        n = len(df)
        k = len(groups)
        eta_squared = f_stat * (k - 1) / (f_stat * (k - 1) + (n - k))
        effect_sizes['eta_squared'] = eta_squared
        
    elif var1_type == 'numérique' and var2_type == 'numérique':
        # Coefficient de corrélation de Pearson
        corr_coef, _ = pearsonr(df[var1].dropna(), df[var2].dropna())
        effect_sizes['pearson_r'] = corr_coef
    
    return effect_sizes

def create_multivariate_analysis(df, target_var, feature_vars):
    """
    Effectue une analyse multivariée basique
    """
    results = {}
    
    # Matrice de corrélation avec la variable cible
    numerical_vars = [var for var in feature_vars if df[var].dtype in ['int64', 'float64']]
    if numerical_vars and df[target_var].dtype in ['int64', 'float64']:
        correlations = {}
        for var in numerical_vars:
            corr, _ = pearsonr(df[var].dropna(), df[target_var].dropna())
            correlations[var] = corr
        results['correlations_with_target'] = correlations
    
    # Importance des features (méthode simple)
    if df[target_var].dtype in ['int64', 'float64']:
        # Régression linéaire simple pour chaque variable
        feature_importance = {}
        for var in feature_vars:
            if df[var].dtype in ['int64', 'float64']:
                clean_data = df[[var, target_var]].dropna()
                if len(clean_data) > 1:
                    corr, _ = pearsonr(clean_data[var], clean_data[target_var])
                    feature_importance[var] = abs(corr)
        results['feature_importance'] = feature_importance
    
    return results

def export_analysis_report(df, analyses, filename="rapport_analyse.html"):
    """
    Exporte un rapport d'analyse complet au format HTML
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
        
        <div class="section">
            <h2>📊 Informations Générales</h2>
            <p><strong>Nombre d'observations:</strong> {len(df)}</p>
            <p><strong>Nombre de variables:</strong> {len(df.columns)}</p>
            <p><strong>Variables numériques:</strong> {len(df.select_dtypes(include=[np.number]).columns)}</p>
            <p><strong>Variables catégorielles:</strong> {len(df.select_dtypes(include=['object']).columns)}</p>
        </div>
    """
    
    # Ajouter les analyses spécifiques
    for analysis_name, analysis_data in analyses.items():
        html_content += f"""
        <div class="section">
            <h2>📈 {analysis_name}</h2>
            <pre>{str(analysis_data)}</pre>
        </div>
        """
    
    html_content += f"""
        <div class="footer">
            <p>🔬 Powered by <strong>Lab_Math SCSM</strong> and <strong>CIE</strong> | © Copyright 2025</p>
            <p>Rappart généré automatiquement par LabMath Analytics Pro</p>
        </div>
    </body>
    </html>
    """
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return filename