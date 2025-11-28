import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

class DataGenerator:
    """
    Classe pour générer des données d'exemple réalistes
    """
    
    def __init__(self):
        self.fake = Faker('fr_FR')
        np.random.seed(42)
        random.seed(42)
    
    def generate_complex_dataset(self, n_observations=1000, n_categorical=5, 
                               n_numerical=7, n_binary=3, missing_percentage=5.0):
        """
        Génère un dataset complexe avec différents types de variables
        """
        data = {}
        
        # Variables catégorielles
        categorical_vars = self._generate_categorical_variables(n_categorical, n_observations)
        data.update(categorical_vars)
        
        # Variables numériques
        numerical_vars = self._generate_numerical_variables(n_numerical, n_observations)
        data.update(numerical_vars)
        
        # Variables binaires
        binary_vars = self._generate_binary_variables(n_binary, n_observations)
        data.update(binary_vars)
        
        # Variable d'intérêt (cible)
        data['Var_Interet'] = self._generate_target_variable(data, n_observations)
        
        # Créer le DataFrame
        df = pd.DataFrame(data)
        
        # Ajouter des valeurs manquantes
        if missing_percentage > 0:
            df = self._add_missing_values(df, missing_percentage)
        
        return df
    
    def _generate_categorical_variables(self, n_vars, n_obs):
        """
        Génère des variables catégorielles réalistes
        """
        vars_dict = {}
        
        # Catégories prédéfinies pour plus de réalisme
        categories = {
            'Region': ['Nord', 'Sud', 'Est', 'Ouest', 'Centre'],
            'Type_Etablissement': ['Hôpital', 'Clinique', 'Laboratoire', 'Centre de santé', 'Dispensaire'],
            'Niveau_Complexite': ['Level I', 'Level II', 'Level III', 'Level IV'],
            'Specialite': ['Généraliste', 'Cardiologie', 'Pédiatrie', 'Chirurgie', 'Urgence'],
            'Statut': ['Public', 'Privé', 'Mixte'],
            'Zone': ['Urbaine', 'Rurale', 'Périurbaine'],
            'Accreditation': ['Oui', 'Non', 'En cours'],
            'Equipement': ['Basique', 'Intermédiaire', 'Avancé'],
            'Personnel': ['Insuffisant', 'Adéquat', 'Abondant'],
            'Financement': ['Etat', 'Privé', 'International', 'Mixte']
        }
        
        category_keys = list(categories.keys())
        
        for i in range(n_vars):
            if i < len(category_keys):
                var_name = category_keys[i]
                categories_list = categories[var_name]
            else:
                var_name = f"Cat_Var_{i+1}"
                categories_list = [f'Cat_{j}' for j in range(random.randint(3, 8))]
            
            vars_dict[var_name] = np.random.choice(
                categories_list, 
                n_obs,
                p=[1/len(categories_list)] * len(categories_list)
            )
        
        return vars_dict
    
    def _generate_numerical_variables(self, n_vars, n_obs):
        """
        Génère des variables numériques réalistes
        """
        vars_dict = {}
        
        # Distributions variées pour plus de réalisme
        numerical_configs = [
            {'name': 'Age_Patients', 'dist': 'normal', 'params': [45, 15], 'min': 18, 'max': 90},
            {'name': 'Nombre_Lits', 'dist': 'poisson', 'params': [50], 'min': 10, 'max': 200},
            {'name': 'Budget_Annuel', 'dist': 'lognormal', 'params': [12, 1.5], 'min': 50000, 'max': 5000000},
            {'name': 'Personnel_Medical', 'dist': 'normal', 'params': [25, 10], 'min': 5, 'max': 100},
            {'name': 'Patients_Jour', 'dist': 'poisson', 'params': [30], 'min': 5, 'max': 100},
            {'name': 'Taux_Occupation', 'dist': 'beta', 'params': [2, 2], 'min': 0.3, 'max': 0.95},
            {'name': 'Distance_Hopital', 'dist': 'exponential', 'params': [0.1], 'min': 0, 'max': 50},
            {'name': 'Satisfaction_Patients', 'dist': 'normal', 'params': [7.5, 1.5], 'min': 1, 'max': 10},
            {'name': 'Duree_Sejour', 'dist': 'gamma', 'params': [2, 2], 'min': 1, 'max': 30},
            {'name': 'Cout_Operation', 'dist': 'lognormal', 'params': [8, 1], 'min': 100, 'max': 10000}
        ]
        
        for i in range(n_vars):
            if i < len(numerical_configs):
                config = numerical_configs[i]
                var_name = config['name']
                
                if config['dist'] == 'normal':
                    values = np.random.normal(config['params'][0], config['params'][1], n_obs)
                elif config['dist'] == 'poisson':
                    values = np.random.poisson(config['params'][0], n_obs)
                elif config['dist'] == 'lognormal':
                    values = np.random.lognormal(config['params'][0], config['params'][1], n_obs)
                elif config['dist'] == 'beta':
                    values = np.random.beta(config['params'][0], config['params'][1], n_obs)
                elif config['dist'] == 'exponential':
                    values = np.random.exponential(config['params'][0], n_obs)
                elif config['dist'] == 'gamma':
                    values = np.random.gamma(config['params'][0], config['params'][1], n_obs)
                else:
                    values = np.random.normal(0, 1, n_obs)
                
                # Appliquer les limites
                values = np.clip(values, config['min'], config['max'])
                
            else:
                var_name = f"Num_Var_{i+1}"
                values = np.random.normal(0, 1, n_obs)
                values = np.round(values, 2)
            
            vars_dict[var_name] = values
        
        return vars_dict
    
    def _generate_binary_variables(self, n_vars, n_obs):
        """
        Génère des variables binaires
        """
        vars_dict = {}
        
        binary_configs = [
            {'name': 'Urgence_Disponible', 'p': 0.7},
            {'name': 'Laboratoire_Interne', 'p': 0.6},
            {'name': 'Radiologie', 'p': 0.5},
            {'name': 'Pharmacy', 'p': 0.8},
            {'name': 'Ambulance', 'p': 0.4},
            {'name': 'Bloc_Operatoire', 'p': 0.3},
            {'name': 'Soins_Intensifs', 'p': 0.2}
        ]
        
        for i in range(n_vars):
            if i < len(binary_configs):
                config = binary_configs[i]
                var_name = config['name']
                p = config['p']
            else:
                var_name = f"Bin_Var_{i+1}"
                p = random.uniform(0.2, 0.8)
            
            vars_dict[var_name] = np.random.choice([0, 1], n_obs, p=[1-p, p])
        
        return vars_dict
    
    def _generate_target_variable(self, data, n_obs):
        """
        Génère une variable cible corrélée avec d'autres variables
        """
        # Créer une variable cible basée sur une combinaison linéaire
        target = np.zeros(n_obs)
        
        # Ajouter de l'aléatoire
        target += np.random.normal(0, 1, n_obs)
        
        # Ajouter des corrélations avec certaines variables numériques
        numerical_keys = [k for k in data.keys() if isinstance(data[k], np.ndarray) and data[k].dtype in [np.float64, np.int64]]
        
        for i, key in enumerate(numerical_keys[:3]):  # Utiliser les 3 premières variables numériques
            if len(data[key]) == n_obs:
                target += 0.3 * (data[key] - np.mean(data[key])) / np.std(data[key])
        
        # Convertir en variable catégorielle pour la classification
        quartiles = np.percentile(target, [25, 50, 75])
        target_cat = np.digitize(target, quartiles)
        categories = ['Faible', 'Moyen', 'Élevé', 'Très élevé']
        
        return [categories[min(i, 3)] for i in target_cat]
    
    def _add_missing_values(self, df, percentage):
        """
        Ajoute des valeurs manquantes aléatoires
        """
        df_with_na = df.copy()
        n_missing = int(len(df) * len(df.columns) * percentage / 100)
        
        for _ in range(n_missing):
            col = np.random.choice(df.columns)
            row = np.random.randint(0, len(df))
            df_with_na.loc[row, col] = np.nan
        
        return df_with_na

    # =========================================================================
    # NOUVELLES FONCTIONS POUR LES TABLEAUX DE CONTINGENCE CORRIGÉS
    # =========================================================================
    
    def generer_tableau_contingence_corrige(self, df, variable_ligne, variable_colonne, pourcentage_type='total'):
        """
        Génère un tableau de contingence avec les formules statistiques CORRECTES
        
        Formules utilisées :
        - n.. : effectif total
        - nij : effectif de la cellule (i,j)
        - ni. : effectif total de la ligne i  
        - n.j : effectif total de la colonne j
        
        Types de pourcentages :
        - 'total'    : pij = nij / n.. × 100  (fréquences conjointes)
        - 'ligne'    : pij = nij / ni. × 100  (profil ligne)
                      fi. = ni. / n.. × 100  (totaux ligne)
                      f.j = n.j / n.. × 100  (totaux colonne)
        - 'colonne'  : pij = nij / n.j × 100  (profil colonne)
                      fi. = ni. / n.. × 100  (totaux ligne)
                      f.j = n.j / n.. × 100  (totaux colonne)
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
            # POURCENTAGES LIGNE: 
            # Cellules internes: pij = nij / ni. × 100
            # Totaux ligne: fi. = ni. / n.. × 100  
            # Totaux colonne: f.j = n.j / n.. × 100
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
            # POURCENTAGES COLONNE:
            # Cellules internes: pij = nij / n.j × 100
            # Totaux ligne: fi. = ni. / n.. × 100
            # Totaux colonne: f.j = n.j / n.. × 100
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

    def afficher_formules_statistiques(self):
        """
        Affiche les formules statistiques utilisées pour plus de transparence
        """
        formules = """
        📊 **FORMULES STATISTIQUES DES TABLEAUX DE CONTINGENCE**

        **Notations:**
        - n.. : Effectif total de la population
        - nij : Effectif de la cellule (ligne i, colonne j)  
        - ni. : Effectif total de la ligne i
        - n.j : Effectif total de la colonne j

        **Types de pourcentages disponibles:**

        🟦 **POURCENTAGES TOTAUX (fréquences conjointes)**
        • Formule: pij = nij / n.. × 100
        • Interprétation: Pourcentage par rapport au total général

        🟩 **POURCENTAGES LIGNE (profil ligne)**
        • Cellules: pij = nij / ni. × 100
        • Totaux ligne: fi. = ni. / n.. × 100  
        • Totaux colonne: f.j = n.j / n.. × 100
        • Interprétation: Pourcentage par rapport au total de la ligne

        🟨 **POURCENTAGES COLONNE (profil colonne)**
        • Cellules: pij = nij / n.j × 100
        • Totaux ligne: fi. = ni. / n.. × 100
        • Totaux colonne: f.j = n.j / n.. × 100
        • Interprétation: Pourcentage par rapport au total de la colonne

        ✅ **GARANTIE:** Tous les calculs respectent ces formules statistiques
        """
        return formules

# Fonction utilitaire pour Streamlit
def creer_interface_tableaux_contingence(df):
    """
    Crée une interface Streamlit pour les tableaux de contingence
    """
    import streamlit as st
    import io
    
    st.header("📊 Tableaux de Contingence - Version Corrigée")
    
    # Afficher les formules
    with st.expander("📖 Voir les formules statistiques utilisées"):
        generateur = DataGenerator()
        st.markdown(generateur.afficher_formules_statistiques())
    
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
            'total': '🟦 Pourcentages totaux (pij = nij/n.. × 100)',
            'ligne': '🟩 Pourcentages ligne (pij = nij/ni. × 100)',
            'colonne': '🟨 Pourcentages colonne (pij = nij/n.j × 100)'
        }[x],
        horizontal=True
    )
    
    # Génération du tableau
    if st.button("🔄 Générer le tableau corrigé", type="primary"):
        generateur = DataGenerator()
        
        with st.spinner("Calcul en cours..."):
            tableau = generateur.generer_tableau_contingence_corrige(
                df, variable_ligne, variable_colonne, type_pourcentage
            )
        
        st.success("✅ Tableau généré avec les formules statistiques correctes!")
        st.dataframe(tableau, use_container_width=True)
        
        # Téléchargement
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

# Exemple d'utilisation
if __name__ == "__main__":
    # Test des fonctions
    generateur = DataGenerator()
    df_test = generateur.generate_complex_dataset(100)
    
    print("Test des tableaux de contingence corrigés:")
    tableau_test = generateur.generer_tableau_contingence_corrige(
        df_test, 'Type_Etablissement', 'Niveau_Complexite', 'ligne'
    )
    print(tableau_test)
    
    print("\n" + "="*50)
    print(generateur.afficher_formules_statistiques())