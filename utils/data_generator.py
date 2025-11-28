import pandas as pd
import numpy as np
import streamlit as st
import io

def generer_tableau_contingence_corrige(df, variable_ligne, variable_colonne, pourcentage_type='total'):
    """
    Génère un tableau de contingence avec les bonnes formules statistiques
    
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
    
    # Calculer n.. (effectif total)
    n_total = tableau_effectifs.loc['Total', 'Total']
    
    # Initialiser le tableau des pourcentages
    tableau_pourcentages = tableau_effectifs.copy().astype(float)
    
    if pourcentage_type == 'total':
        # POURCENTAGES TOTAUX: pij = nij / n.. × 100 pour TOUTES les cellules
        tableau_pourcentages = (tableau_effectifs / n_total * 100).round(1)
        
    elif pourcentage_type == 'ligne':
        # POURCENTAGES LIGNE: pij = nij / ni. × 100 pour les cellules internes
        # Pour les totaux: fi. = ni. / n.. × 100 et f.j = n.j / n.. × 100
        for i, idx in enumerate(tableau_effectifs.index):
            for j, col in enumerate(tableau_effectifs.columns):
                nij = tableau_effectifs.iloc[i, j]
                
                if idx == 'Total' and col == 'Total':
                    # Coin inférieur droit: 100%
                    tableau_pourcentages.iloc[i, j] = 100.0
                elif idx == 'Total':
                    # Totaux colonne: f.j = n.j / n.. × 100
                    n_j = tableau_effectifs.loc['Total', col]
                    tableau_pourcentages.iloc[i, j] = (n_j / n_total * 100).round(1)
                elif col == 'Total':
                    # Totaux ligne: fi. = ni. / n.. × 100
                    n_i = tableau_effectifs.loc[idx, 'Total']
                    tableau_pourcentages.iloc[i, j] = (n_i / n_total * 100).round(1)
                else:
                    # Cellules internes: pij = nij / ni. × 100
                    n_i = tableau_effectifs.loc[idx, 'Total']
                    if n_i > 0:
                        tableau_pourcentages.iloc[i, j] = (nij / n_i * 100).round(1)
                    else:
                        tableau_pourcentages.iloc[i, j] = 0.0
                        
    elif pourcentage_type == 'colonne':
        # POURCENTAGES COLONNE: pij = nij / n.j × 100 pour les cellules internes
        # Pour les totaux: fi. = ni. / n.. × 100 et f.j = n.j / n.. × 100
        for i, idx in enumerate(tableau_effectifs.index):
            for j, col in enumerate(tableau_effectifs.columns):
                nij = tableau_effectifs.iloc[i, j]
                
                if idx == 'Total' and col == 'Total':
                    # Coin inférieur droit: 100%
                    tableau_pourcentages.iloc[i, j] = 100.0
                elif idx == 'Total':
                    # Totaux colonne: f.j = n.j / n.. × 100
                    n_j = tableau_effectifs.loc['Total', col]
                    tableau_pourcentages.iloc[i, j] = (n_j / n_total * 100).round(1)
                elif col == 'Total':
                    # Totaux ligne: fi. = ni. / n.. × 100
                    n_i = tableau_effectifs.loc[idx, 'Total']
                    tableau_pourcentages.iloc[i, j] = (n_i / n_total * 100).round(1)
                else:
                    # Cellules internes: pij = nij / n.j × 100
                    n_j = tableau_effectifs.loc['Total', col]
                    if n_j > 0:
                        tableau_pourcentages.iloc[i, j] = (nij / n_j * 100).round(1)
                    else:
                        tableau_pourcentages.iloc[i, j] = 0.0
    
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
        'ligne': "Cellules: pij = nij/ni. × 100 | Totaux: fi. = ni./n.. × 100, f.j = n.j/n.. × 100",
        'colonne': "Cellules: pij = nij/n.j × 100 | Totaux: fi. = ni./n.. × 100, f.j = n.j/n.. × 100"
    }
    
    st.subheader(titres[type_pourcentage])
    st.dataframe(tableau, use_container_width=True)
    st.caption(f"📊 **Formules utilisées** : {formules[type_pourcentage]}")
    
    return tableau

# Test avec vos données exactes - POUR VERIFICATION
def tester_avec_vos_donnees_corrige():
    """
    Test avec les données exactes de votre exemple pour vérifier les calculs
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
    
    st.write("## 🧪 TEST AVEC VOS DONNÉES EXACTES (POURCENTAGES LIGNE)")
    
    # Tableau avec pourcentages ligne (comme dans votre exemple)
    st.write("### 📋 TABLEAU CORRIGÉ (Pourcentages ligne)")
    tableau_corrige = generer_tableau_contingence_corrige(
        df_exact, 'Type_Etablissement', 'Niveau_Complexite', 'ligne'
    )
    
    st.dataframe(tableau_corrige, use_container_width=True)
    
    # VÉRIFICATION DES CALCULS
    st.write("### 🔍 VÉRIFICATION DES CALCULS")
    
    n_total = 548  # Total général
    
    st.write("**Calculs des totaux ligne (fi. = ni./n.. × 100):**")
    st.write(f"- confessionnel : 57 / 548 × 100 = {(57/548*100):.1f}%")
    st.write(f"- private : 86 / 548 × 100 = {(86/548*100):.1f}%") 
    st.write(f"- public : 405 / 548 × 100 = {(405/548*100):.1f}%")
    
    st.write("**Calculs des totaux colonne (f.j = n.j/n.. × 100):**")
    st.write(f"- Level I : 284 / 548 × 100 = {(284/548*100):.1f}%")
    st.write(f"- Level II : 211 / 548 × 100 = {(211/548*100):.1f}%")
    st.write(f"- Level III : 32 / 548 × 100 = {(32/548*100):.1f}%")
    st.write(f"- Level IV : 21 / 548 × 100 = {(21/548*100):.1f}%")
    
    st.write("**Calculs des cellules internes (pij = nij/ni. × 100):**")
    st.write(f"- confessionnel, Level I : 20 / 57 × 100 = {(20/57*100):.1f}%")
    st.write(f"- public, Level III : 30 / 405 × 100 = {(30/405*100):.1f}%")
    
    return tableau_corrige

# Interface Streamlit complète
def interface_analyse_contingence_corrigee(df):
    """
    Interface complète pour l'analyse des tableaux de contingence CORRIGÉE
    """
    st.header("📊 Analyse des Tableaux de Contingence (Version Finale Corrigée)")
    
    # Sélection des variables
    col1, col2 = st.columns(2)
    
    with col1:
        variable_ligne = st.selectbox(
            "Variable pour les lignes:",
            options=df.columns,
            index=0,
            key="var_ligne_contingence"
        )
    
    with col2:
        variable_colonne = st.selectbox(
            "Variable pour les colonnes:", 
            options=df.columns,
            index=1 if len(df.columns) > 1 else 0,
            key="var_colonne_contingence"
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
    if st.button("🔄 Générer le tableau corrigé", type="primary", key="btn_contingence"):
        
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
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_contingence"
        )
    
    # Section test avec vos données
    with st.expander("🧪 Tester avec les données de l'exemple fourni"):
        if st.button("Tester avec les données exactes de l'exemple", key="btn_test_exemple"):
            tester_avec_vos_donnees_corrige()

# Pour intégration dans votre app principale
def ajouter_section_contingence_dans_app(df):
    """
    À ajouter dans votre fonction main() de Streamlit
    """
    if st.sidebar.checkbox("📊 Tableaux de contingence (CORRIGÉ)", key="check_contingence"):
        interface_analyse_contingence_corrigee(df)

# Exemple d'utilisation
if __name__ == "__main__":
    # Test local
    data_test = {
        'Type_Etablissement': 
            ['public'] * 234 + ['private'] * 30 + ['confessionnel'] * 20 + 
            ['public'] * 120 + ['private'] * 56 + ['confessionnel'] * 35 +
            ['public'] * 30 + ['confessionnel'] * 2 +
            ['public'] * 21,
        'Niveau_Complexite': 
            ['Level I'] * 284 + ['Level II'] * 211 + ['Level III'] * 32 + ['Level IV'] * 21
    }
    
    df_test = pd.DataFrame(data_test)
    interface_analyse_contingence_corrigee(df_test)