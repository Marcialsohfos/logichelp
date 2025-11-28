import pandas as pd
import numpy as np
import streamlit as st
import io

def generer_tableau_contingence_corrige(df, variable_ligne, variable_colonne, pourcentage_type='total'):
    """
    Génère un tableau de contingence COMPLÈTEMENT corrigé avec les bonnes formules statistiques
    
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
    pd.DataFrame : Tableau de contingence formaté avec tous les pourcentages corrects
    """
    
    # Créer le tableau de contingence avec les effectifs
    tableau_effectifs = pd.crosstab(
        df[variable_ligne], 
        df[variable_colonne],
        margins=True,
        margins_name='Total'
    )
    
    # Calculer n.. (effectif total)
    n_total = tableau_effectifs.loc['Total', 'Total']
    
    # Calculer les pourcentages selon le type choisi
    if pourcentage_type == 'total':
        # Pourcentages par rapport au total général (fréquences conjointes)
        # pij = nij / n.. × 100 pour toutes les cellules
        tableau_pourcentages = (tableau_effectifs / n_total * 100).round(1)
        
    elif pourcentage_type == 'ligne':
        # Pourcentages par ligne (profil ligne)
        # pij = nij / ni. × 100 pour les cellules internes
        # Pour les totaux colonne : n.j / n.. × 100
        tableau_pourcentages = (tableau_effectifs.div(tableau_effectifs.sum(axis=1), axis=0) * 100).round(1)
        # Corriger la dernière ligne (totaux)
        for col in tableau_effectifs.columns:
            if col != 'Total':
                n_j = tableau_effectifs.loc['Total', col]
                tableau_pourcentages.loc['Total', col] = (n_j / n_total * 100).round(1)
        
    elif pourcentage_type == 'colonne':
        # Pourcentages par colonne (profil colonne)
        # pij = nij / n.j × 100 pour les cellules internes
        # Pour les totaux ligne : ni. / n.. × 100
        tableau_pourcentages = (tableau_effectifs.div(tableau_effectifs.sum(axis=0), axis=1) * 100).round(1)
        # Corriger la dernière colonne (totaux)
        for idx in tableau_effectifs.index:
            if idx != 'Total':
                n_i = tableau_effectifs.loc[idx, 'Total']
                tableau_pourcentages.loc[idx, 'Total'] = (n_i / n_total * 100).round(1)
    
    else:
        raise ValueError("Type de pourcentage doit être 'total', 'ligne' ou 'colonne'")
    
    # Pour le total général (coin inférieur droit) : toujours 100%
    tableau_pourcentages.loc['Total', 'Total'] = 100.0
    
    # Combiner effectifs et pourcentages
    tableau_final = tableau_effectifs.copy().astype(object)
    
    for i in range(tableau_effectifs.shape[0]):
        for j in range(tableau_effectifs.shape[1]):
            effectif = tableau_effectifs.iloc[i, j]
            pourcentage = tableau_pourcentages.iloc[i, j]
            
            if pd.notna(effectif) and pd.notna(pourcentage):
                tableau_final.iloc[i, j] = f"{effectif} ({pourcentage}%)"
            else:
                tableau_final.iloc[i, j] = "0 (0.0%)"
    
    return tableau_final

def afficher_tableau_contingence_streamlit(df, variable_ligne, variable_colonne, type_pourcentage='total'):
    """
    Affiche un tableau de contingence dans Streamlit avec les bonnes formules
    """
    tableau = generer_tableau_contingence_corrige(df, variable_ligne, variable_colonne, type_pourcentage)
    
    # Titre selon le type de pourcentage
    titres = {
        'total': f"Répartition des {variable_ligne} selon {variable_colonne} - Pourcentages totaux",
        'ligne': f"Répartition des {variable_ligne} selon {variable_colonne} - Pourcentages ligne", 
        'colonne': f"Répartition des {variable_ligne} selon {variable_colonne} - Pourcentages colonne"
    }
    
    formules = {
        'total': "pij = nij/n.. × 100 (fréquences conjointes)",
        'ligne': "pij = nij/ni. × 100 (profil ligne), totaux colonne: n.j/n.. × 100",
        'colonne': "pij = nij/n.j × 100 (profil colonne), totaux ligne: ni./n.. × 100"
    }
    
    st.subheader(titres[type_pourcentage])
    st.dataframe(tableau, use_container_width=True)
    st.caption(f"📊 **Formules utilisées** : {formules[type_pourcentage]}")
    
    return tableau

# Fonction pour tester avec vos données exactes
def tester_avec_vos_donnees():
    """
    Test avec les données exactes de votre exemple
    """
    # Recréer exactement vos données
    data_exact = {
        'Type_Etablissement': 
            ['public'] * 234 + ['private'] * 30 + ['confessionnel'] * 20 +  # Level I
            ['public'] * 120 + ['private'] * 56 + ['confessionnel'] * 35 +   # Level II  
            ['public'] * 30 + ['confessionnel'] * 2 +                        # Level III
            ['public'] * 21,                                                 # Level IV
        'Niveau_Complexite': 
            ['Level I'] * 284 + 
            ['Level II'] * 211 + 
            ['Level III'] * 32 + 
            ['Level IV'] * 21
    }
    
    df_exact = pd.DataFrame(data_exact)
    
    st.write("## 🧪 TEST AVEC VOS DONNÉES EXACTES")
    
    # Tableau avec pourcentages totaux
    st.write("### 📋 VOTRE TABLEAU CORRIGÉ (Pourcentages totaux)")
    tableau_corrige = afficher_tableau_contingence_streamlit(
        df_exact, 'Type_Etablissement', 'Niveau_Complexite', 'total'
    )
    
    # Vérification des calculs
    st.write("### 🔍 VÉRIFICATION DES CALCULS")
    
    n_total = 548  # Total général
    
    # Vérification Level I total
    n_level1 = 284
    pourcent_level1_attendu = (284 / 548 * 100)
    st.write(f"**Level I total** : {n_level1} / {n_total} × 100 = {pourcent_level1_attendu:.1f}%")
    
    # Vérification Public total  
    n_public = 405
    pourcent_public_attendu = (405 / 548 * 100)
    st.write(f"**Public total** : {n_public} / {n_total} × 100 = {pourcent_public_attendu:.1f}%")
    
    return tableau_corrige

# Interface Streamlit complète pour l'analyse de contingence
def interface_analyse_contingence(df):
    """
    Interface complète pour l'analyse des tableaux de contingence
    """
    st.header("📊 Analyse des Tableaux de Contingence (Version Corrigée)")
    
    # Sélection des variables
    col1, col2 = st.columns(2)
    
    with col1:
        variable_ligne = st.selectbox(
            "Variable pour les lignes:",
            options=df.columns,
            index=0,
            key="var_ligne"
        )
    
    with col2:
        variable_colonne = st.selectbox(
            "Variable pour les colonnes:", 
            options=df.columns,
            index=1 if len(df.columns) > 1 else 0,
            key="var_colonne"
        )
    
    # Type de pourcentage
    type_pourcentage = st.radio(
        "**Type de pourcentage**:",
        options=['total', 'ligne', 'colonne'],
        format_func=lambda x: {
            'total': '🟦 Pourcentages totaux (pij = nij/n.. × 100)',
            'ligne': '🟩 Pourcentages ligne (pij = nij/ni. × 100)',
            'colonne': '🟨 Pourcentages colonne (pij = nij/n.j × 100)'
        }[x],
        horizontal=True
    )
    
    # Bouton de génération
    if st.button("🔄 Générer le tableau corrigé", type="primary"):
        
        # Afficher le tableau
        tableau = afficher_tableau_contingence_streamlit(
            df, variable_ligne, variable_colonne, type_pourcentage
        )
        
        # Option de téléchargement
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            tableau.to_excel(writer, sheet_name='Tableau_Contingence', index=True)
        
        output.seek(0)
        
        st.download_button(
            label="📥 Télécharger le tableau Excel",
            data=output,
            file_name=f"tableau_contingence_{variable_ligne}_{variable_colonne}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # Section test avec vos données
    with st.expander("🧪 Tester avec les données de l'exemple"):
        if st.button("Tester avec les données de l'exemple fourni"):
            tester_avec_vos_donnees()

# Exemple d'utilisation dans votre app principale
"""
Dans votre fonction main() de Streamlit, ajoutez:

if st.sidebar.checkbox("📊 Tableaux de contingence (CORRIGÉ)"):
    interface_analyse_contingence(df)
"""

# Fonction utilitaire pour debug
def debug_tableau(df, var_ligne, var_colonne):
    """
    Fonction de debug pour vérifier tous les calculs
    """
    tableau_effectifs = pd.crosstab(df[var_ligne], df[var_colonne], margins=True)
    
    st.write("### 🐛 DEBUG - Calculs détaillés")
    st.write("**Tableau des effectifs:**")
    st.dataframe(tableau_effectifs)
    
    n_total = tableau_effectifs.loc['Total', 'Total']
    st.write(f"n.. (total général) = {n_total}")
    
    st.write("**Vérification des totaux:**")
    for idx in tableau_effectifs.index:
        if idx != 'Total':
            n_i = tableau_effectifs.loc[idx, 'Total']
            pourcent_i = (n_i / n_total * 100)
            st.write(f"- {idx} : {n_i} / {n_total} × 100 = {pourcent_i:.1f}%")