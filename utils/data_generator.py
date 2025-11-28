import pandas as pd
import numpy as np

def generer_tableau_contingence_corrige(df, variable_ligne, variable_colonne, pourcentage_type='total'):
    """
    Génère un tableau de contingence corrigé avec les bonnes formules statistiques
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données
    variable_ligne : str
        Variable pour les lignes du tableau
    variable_colonne : str
        Variable pour les colonnes du tableau
    pourcentage_type : str
        Type de pourcentage : 'total', 'ligne', 'colonne'
    
    Returns:
    --------
    pd.DataFrame : Tableau de contingence formaté
    """
    
    # Créer le tableau de contingence avec les effectifs
    tableau_effectifs = pd.crosstab(
        df[variable_ligne], 
        df[variable_colonne],
        margins=True,
        margins_name='Total'
    )
    
    # Calculer les pourcentages selon le type choisi
    n_total = tableau_effectifs.loc['Total', 'Total']  # n.. effectif total
    
    if pourcentage_type == 'total':
        # Pourcentages par rapport au total général (fréquences conjointes)
        tableau_pourcentages = (tableau_effectifs / n_total * 100).round(1)
    elif pourcentage_type == 'ligne':
        # Pourcentages par ligne (profil ligne)
        tableau_pourcentages = (tableau_effectifs.div(tableau_effectifs.sum(axis=1), axis=0) * 100).round(1)
    elif pourcentage_type == 'colonne':
        # Pourcentages par colonne (profil colonne)
        tableau_pourcentages = (tableau_effectifs.div(tableau_effectifs.sum(axis=0), axis=1) * 100).round(1)
    else:
        raise ValueError("Type de pourcentage doit être 'total', 'ligne' ou 'colonne'")
    
    # Combiner effectifs et pourcentages
    tableau_final = tableau_effectifs.copy().astype(object)
    
    for i in range(tableau_effectifs.shape[0]):
        for j in range(tableau_effectifs.shape[1]):
            effectif = tableau_effectifs.iloc[i, j]
            pourcentage = tableau_pourcentages.iloc[i, j]
            
            if pd.notna(effectif) and pd.notna(pourcentage):
                if i == tableau_effectifs.shape[0]-1 or j == tableau_effectifs.shape[1]-1:
                    # Pour les totaux, afficher seulement l'effectif
                    tableau_final.iloc[i, j] = f"{effectif}"
                else:
                    tableau_final.iloc[i, j] = f"{effectif} ({pourcentage}%)"
            else:
                tableau_final.iloc[i, j] = "0 (0.0%)"
    
    return tableau_final

def generer_tableau_complet(df, variable_ligne, variable_colonne, titre=None):
    """
    Génère un tableau complet avec les trois types de pourcentages
    """
    if titre is None:
        titre = f"Répartition des {variable_ligne} selon {variable_colonne}"
    
    # Tableau avec pourcentages totaux (fréquences conjointes)
    tableau_total = generer_tableau_contingence_corrige(df, variable_ligne, variable_colonne, 'total')
    
    # Tableau avec pourcentages ligne
    tableau_ligne = generer_tableau_contingence_corrige(df, variable_ligne, variable_colonne, 'ligne')
    
    # Tableau avec pourcentages colonne
    tableau_colonne = generer_tableau_contingence_corrige(df, variable_ligne, variable_colonne, 'colonne')
    
    return {
        'total': tableau_total,
        'ligne': tableau_ligne,
        'colonne': tableau_colonne,
        'titre': titre
    }

# Fonction utilitaire pour afficher les tableaux dans Streamlit
def afficher_tableau_contingence_streamlit(df, variable_ligne, variable_colonne, type_pourcentage='total'):
    """
    Affiche un tableau de contingence dans Streamlit
    """
    tableau = generer_tableau_contingence_corrige(df, variable_ligne, variable_colonne, type_pourcentage)
    
    # Déterminer le titre selon le type de pourcentage
    if type_pourcentage == 'total':
        titre = f"Répartition des {variable_ligne} selon {variable_colonne} - Pourcentages totaux"
    elif type_pourcentage == 'ligne':
        titre = f"Répartition des {variable_ligne} selon {variable_colonne} - Pourcentages ligne"
    else:
        titre = f"Répartition des {variable_ligne} selon {variable_colonne} - Pourcentages colonne"
    
    st.subheader(titre)
    st.dataframe(tableau, use_container_width=True)
    
    # Ajouter une explication
    if type_pourcentage == 'total':
        st.caption("📊 **Lecture** : Effectifs (pourcentage du total général) - pij = nij/n.. × 100")
    elif type_pourcentage == 'ligne':
        st.caption("📊 **Lecture** : Effectifs (pourcentage de la ligne) - pij = nij/ni. × 100")
    else:
        st.caption("📊 **Lecture** : Effectifs (pourcentage de la colonne) - pij = nij/n.j × 100")

# Version alternative avec calcul détaillé pour debug
def generer_tableau_detaille(df, variable_ligne, variable_colonne):
    """
    Génère un tableau détaillé avec tous les calculs pour vérification
    """
    # Tableau de base
    tableau_base = pd.crosstab(df[variable_ligne], df[variable_colonne], margins=True)
    
    # Calculs détaillés
    n_total = tableau_base.loc['All', 'All']
    
    st.write("### 🔍 Calculs détaillés du tableau de contingence")
    
    st.write("**Tableau des effectifs (nij):**")
    st.dataframe(tableau_base)
    
    st.write("**Marges ligne (ni.):**")
    st.write(tableau_base.sum(axis=1))
    
    st.write("**Marges colonne (n.j):**")
    st.write(tableau_base.sum(axis=0))
    
    st.write(f"**Effectif total (n..):** {n_total}")
    
    # Tableau corrigé
    st.write("### ✅ Tableau corrigé avec pourcentages totaux")
    tableau_corrige = generer_tableau_contingence_corrige(df, variable_ligne, variable_colonne, 'total')
    st.dataframe(tableau_corrige)

# Test avec vos données d'exemple
def tester_tableau_corrige():
    """
    Teste la fonction avec des données similaires aux vôtres
    """
    # Créer des données de test similaires
    data_test = {
        'Type_Etablissement': ['public'] * 234 + ['private'] * 30 + ['confessionnel'] * 20 + 
                             ['public'] * 120 + ['private'] * 56 + ['confessionnel'] * 35 +
                             ['public'] * 30 + ['confessionnel'] * 2 +
                             ['public'] * 21,
        'Niveau_Complexite': ['Level I'] * 284 + ['Level II'] * 211 + ['Level III'] * 32 + ['Level IV'] * 21
    }
    
    df_test = pd.DataFrame(data_test)
    
    st.write("## 🧪 Test de la fonction corrigée")
    
    # Tableau avec pourcentages totaux (CORRIGÉ)
    tableau_corrige = generer_tableau_contingence_corrige(df_test, 'Type_Etablissement', 'Niveau_Complexite', 'total')
    
    st.write("### 📋 VOTRE TABLEAU CORRIGÉ (avec pourcentages totaux)")
    st.dataframe(tableau_corrige)
    
    # Vérification des calculs
    st.write("### 🔍 Vérification des calculs")
    
    # Calcul manuel pour public, Level I
    n_public_level1 = 234
    n_total = len(df_test)
    pourcentage_calcule = (n_public_level1 / n_total * 100)
    
    st.write(f"Public, Level I: {n_public_level1} / {n_total} × 100 = {pourcentage_calcule:.1f}%")
    
    return tableau_corrige

# Intégration dans votre application Streamlit existante
def ajouter_analyse_contingence_streamlit(df):
    """
    Ajoute une section d'analyse de contingence à votre app Streamlit
    """
    st.header("📊 Analyse des Tableaux de Contingence")
    
    # Sélection des variables
    col1, col2 = st.columns(2)
    
    with col1:
        variable_ligne = st.selectbox(
            "Variable pour les lignes:",
            options=df.columns,
            index=0
        )
    
    with col2:
        variable_colonne = st.selectbox(
            "Variable pour les colonnes:",
            options=df.columns,
            index=1 if len(df.columns) > 1 else 0
        )
    
    # Type de pourcentage
    type_pourcentage = st.radio(
        "Type de pourcentage:",
        options=['total', 'ligne', 'colonne'],
        format_func=lambda x: {
            'total': 'Pourcentages totaux (pij = nij/n.. × 100)',
            'ligne': 'Pourcentages ligne (pij = nij/ni. × 100)',
            'colonne': 'Pourcentages colonne (pij = nij/n.j × 100)'
        }[x],
        horizontal=True
    )
    
    # Générer et afficher le tableau
    if st.button("🔄 Générer le tableau de contingence", type="primary"):
        afficher_tableau_contingence_streamlit(df, variable_ligne, variable_colonne, type_pourcentage)
        
        # Option pour télécharger le tableau
        tableau = generer_tableau_contingence_corrige(df, variable_ligne, variable_colonne, type_pourcentage)
        
        # Convertir en Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            tableau.to_excel(writer, sheet_name='Tableau_Contingence')
        output.seek(0)
        
        st.download_button(
            label="📥 Télécharger le tableau Excel",
            data=output,
            file_name=f"tableau_contingence_{variable_ligne}_{variable_colonne}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# Exemple d'utilisation dans votre main Streamlit
"""
# Dans votre fonction main() de Streamlit, ajoutez:

if st.sidebar.checkbox("📊 Analyse des tableaux de contingence"):
    ajouter_analyse_contingence_streamlit(df)
"""