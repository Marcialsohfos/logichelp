import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# ======================================================================
#  CLASSE PRINCIPALE : GENERATION DE DONNÉES
# ======================================================================

class DataGenerator:
    """
    Classe permettant de générer :
    - datasets complexes (catégorielles, numériques, binaires, cible)
    - tableaux de contingence 100% corrects statistiquement
    """

    def __init__(self):
        self.fake = Faker("fr_FR")
        np.random.seed(42)
        random.seed(42)

    # ==================================================================
    #  GENERATION DU DATASET RÉALISTE
    # ==================================================================

    def generate_complex_dataset(self, n_observations=1000, n_categorical=5,
                                 n_numerical=7, n_binary=3, missing_percentage=5.0):
        """
        Génère un dataset complexe avec différents types de variables
        """
        data = {}

        # Variables catégorielles
        data.update(self._generate_categorical_variables(n_categorical, n_observations))

        # Variables numériques
        data.update(self._generate_numerical_variables(n_numerical, n_observations))

        # Variables binaires
        data.update(self._generate_binary_variables(n_binary, n_observations))

        # Variable cible corrélée
        data["Var_Interet"] = self._generate_target_variable(data, n_observations)

        # Construction du DataFrame
        df = pd.DataFrame(data)

        # Ajout de NA
        if missing_percentage > 0:
            df = self._add_missing_values(df, missing_percentage)

        return df

    # -------------------------------------------------------------------

    def _generate_categorical_variables(self, n_vars, n_obs):
        """
        Génère des variables catégorielles réalistes
        """
        vars_dict = {}

        predefined = {
            "Region": ["Nord", "Sud", "Est", "Ouest", "Centre"],
            "Type_Etablissement": ["Hôpital", "Clinique", "Laboratoire", "Centre de santé", "Dispensaire"],
            "Niveau_Complexite": ["Level I", "Level II", "Level III", "Level IV"],
            "Specialite": ["Généraliste", "Cardiologie", "Pédiatrie", "Chirurgie", "Urgence"],
            "Statut": ["Public", "Privé", "Mixte"],
            "Zone": ["Urbaine", "Rurale", "Périurbaine"],
            "Accreditation": ["Oui", "Non", "En cours"],
            "Equipement": ["Basique", "Intermédiaire", "Avancé"],
            "Personnel": ["Insuffisant", "Adéquat", "Abondant"],
            "Financement": ["Etat", "Privé", "International", "Mixte"]
        }

        keys = list(predefined.keys())

        for i in range(n_vars):
            if i < len(keys):
                name = keys[i]
                categories = predefined[name]
            else:
                name = f"Cat_Var_{i+1}"
                categories = [f"Cat_{j}" for j in range(random.randint(3, 8))]

            vars_dict[name] = np.random.choice(categories, n_obs)

        return vars_dict

    # -------------------------------------------------------------------

    def _generate_numerical_variables(self, n_vars, n_obs):
        """
        Génère des variables numériques réalistes
        """
        configs = [
            {"name": "Age_Patients", "dist": "normal", "params": [45, 15], "min": 18, "max": 90},
            {"name": "Nombre_Lits", "dist": "poisson", "params": [50], "min": 10, "max": 200},
            {"name": "Budget_Annuel", "dist": "lognormal", "params": [12, 1.5], "min": 50000, "max": 5000000},
            {"name": "Personnel_Medical", "dist": "normal", "params": [25, 10], "min": 5, "max": 100},
            {"name": "Patients_Jour", "dist": "poisson", "params": [30], "min": 5, "max": 100},
            {"name": "Taux_Occupation", "dist": "beta", "params": [2, 2], "min": 0.3, "max": 0.95},
            {"name": "Distance_Hopital", "dist": "exponential", "params": [0.1], "min": 0, "max": 50},
            {"name": "Satisfaction_Patients", "dist": "normal", "params": [7.5, 1.5], "min": 1, "max": 10},
            {"name": "Duree_Sejour", "dist": "gamma", "params": [2, 2], "min": 1, "max": 30},
            {"name": "Cout_Operation", "dist": "lognormal", "params": [8, 1], "min": 100, "max": 10000}
        ]

        vars_dict = {}

        for i in range(n_vars):
            if i < len(configs):
                c = configs[i]
                name = c["name"]

                if c["dist"] == "normal":
                    v = np.random.normal(*c["params"], n_obs)
                elif c["dist"] == "poisson":
                    v = np.random.poisson(c["params"][0], n_obs)
                elif c["dist"] == "lognormal":
                    v = np.random.lognormal(*c["params"], n_obs)
                elif c["dist"] == "beta":
                    v = np.random.beta(*c["params"], n_obs)
                elif c["dist"] == "exponential":
                    v = np.random.exponential(c["params"][0], n_obs)
                elif c["dist"] == "gamma":
                    v = np.random.gamma(*c["params"], n_obs)
                else:
                    v = np.random.normal(0, 1, n_obs)

                v = np.clip(v, c["min"], c["max"])

            else:
                name = f"Num_Var_{i+1}"
                v = np.random.normal(0, 1, n_obs)
                v = np.round(v, 2)

            vars_dict[name] = v

        return vars_dict

    # -------------------------------------------------------------------

    def _generate_binary_variables(self, n_vars, n_obs):
        """
        Génère des variables binaires
        """
        configs = [
            {"name": "Urgence_Disponible", "p": 0.7},
            {"name": "Laboratoire_Interne", "p": 0.6},
            {"name": "Radiologie", "p": 0.5},
            {"name": "Pharmacy", "p": 0.8},
            {"name": "Ambulance", "p": 0.4},
            {"name": "Bloc_Operatoire", "p": 0.3},
            {"name": "Soins_Intensifs", "p": 0.2}
        ]

        vars_dict = {}

        for i in range(n_vars):
            if i < len(configs):
                name = configs[i]["name"]
                p = configs[i]["p"]
            else:
                name = f"Bin_Var_{i+1}"
                p = random.uniform(0.2, 0.8)

            vars_dict[name] = np.random.choice([0, 1], n_obs, p=[1 - p, p])

        return vars_dict

    # -------------------------------------------------------------------

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

    # -------------------------------------------------------------------

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

    # ==================================================================
    #  TABLEAUX DE CONTINGENCE — VERSION PROPRE ET CORRIGÉE
    # ==================================================================

    def generer_tableau_contingence_corrige(self, df, variable_ligne, variable_colonne, pourcentage_type="total"):
        """
        Génère un tableau de contingence avec :
            nij    = effectifs
            ni.    = totaux ligne
            n.j    = totaux colonne
            n..    = total général
            + pourcentages correctement calculés
            
        Formules garanties :
        - Pourcentages totaux : pij = nij / n.. × 100
        - Pourcentages ligne : pij = nij / ni. × 100 (cellules), fi. = ni. / n.. × 100 (totaux ligne), f.j = n.j / n.. × 100 (totaux colonne)
        - Pourcentages colonne : pij = nij / n.j × 100 (cellules), fi. = ni. / n.. × 100 (totaux ligne), f.j = n.j / n.. × 100 (totaux colonne)
        """

        # Tableau brut avec marges
        tab = pd.crosstab(df[variable_ligne], df[variable_colonne], margins=True, margins_name="Total")

        # Récupérer les totaux
        n_total = tab.loc["Total", "Total"]  # n..
        ni = tab["Total"]  # totaux ligne (ni.)
        nj = tab.loc["Total"]  # totaux colonne (n.j)

        # Tableau de pourcentages
        pct = pd.DataFrame(index=tab.index, columns=tab.columns, dtype=float)

        if pourcentage_type == "total":
            # POURCENTAGES TOTAUX : pij = nij / n.. × 100
            pct = (tab / n_total * 100).round(1)

        elif pourcentage_type == "ligne":
            # POURCENTAGES LIGNE : 
            # Cellules : pij = nij / ni. × 100
            # Totaux ligne : fi. = ni. / n.. × 100
            # Totaux colonne : f.j = n.j / n.. × 100
            
            # Calcul des cellules internes
            for idx in tab.index:
                for col in tab.columns:
                    if idx != "Total" and col != "Total":
                        nij = tab.loc[idx, col]
                        ni_val = ni.loc[idx]
                        if ni_val > 0:
                            pct.loc[idx, col] = (nij / ni_val * 100).round(1)
                        else:
                            pct.loc[idx, col] = 0.0
            
            # Totaux ligne
            for idx in tab.index:
                if idx != "Total":
                    pct.loc[idx, "Total"] = (ni.loc[idx] / n_total * 100).round(1)
            
            # Totaux colonne  
            for col in tab.columns:
                if col != "Total":
                    pct.loc["Total", col] = (nj.loc[col] / n_total * 100).round(1)
            
            # Coin inférieur droit
            pct.loc["Total", "Total"] = 100.0

        elif pourcentage_type == "colonne":
            # POURCENTAGES COLONNE :
            # Cellules : pij = nij / n.j × 100
            # Totaux ligne : fi. = ni. / n.. × 100
            # Totaux colonne : f.j = n.j / n.. × 100
            
            # Calcul des cellules internes
            for idx in tab.index:
                for col in tab.columns:
                    if idx != "Total" and col != "Total":
                        nij = tab.loc[idx, col]
                        nj_val = nj.loc[col]
                        if nj_val > 0:
                            pct.loc[idx, col] = (nij / nj_val * 100).round(1)
                        else:
                            pct.loc[idx, col] = 0.0
            
            # Totaux ligne
            for idx in tab.index:
                if idx != "Total":
                    pct.loc[idx, "Total"] = (ni.loc[idx] / n_total * 100).round(1)
            
            # Totaux colonne
            for col in tab.columns:
                if col != "Total":
                    pct.loc["Total", col] = (nj.loc[col] / n_total * 100).round(1)
            
            # Coin inférieur droit
            pct.loc["Total", "Total"] = 100.0

        # Fusion effectif + pourcentage
        final = tab.copy().astype(object)
        for i in tab.index:
            for j in tab.columns:
                n = int(tab.loc[i, j])
                p = pct.loc[i, j]
                final.loc[i, j] = f"{n} ({p}%)"

        return final

    # ==================================================================
    # FORMULES STATISTIQUES
    # ==================================================================

    def afficher_formules_statistiques(self):
        """
        Retourne les formules statistiques utilisées pour la transparence
        """
        return """
        📊 **FORMULES STATISTIQUES UTILISÉES**

        Notations :
        - n.. = effectif total
        - nij = effectif cellule i,j
        - ni. = total ligne i
        - n.j = total colonne j

        **Pourcentages disponibles :**

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

        ✅ **GARANTIE :** Tous les calculs respectent rigoureusement ces formules statistiques
        Les pourcentages partiels (fi. et f.j) sont TOUJOURS calculés par rapport au total général n..
        """


# ======================================================================
# INTERFACE STREAMLIT OPTIONNELLE
# ======================================================================

def creer_interface_tableaux_contingence(df):
    """
    Crée une interface Streamlit pour les tableaux de contingence corrigés
    """
    import streamlit as st
    import io

    st.header("📊 Tableaux de Contingence — Version Corrigée")

    gen = DataGenerator()

    with st.expander("📘 Formules statistiques utilisées"):
        st.markdown(gen.afficher_formules_statistiques())

    col1, col2 = st.columns(2)
    with col1:
        variable_ligne = st.selectbox("Variable pour les lignes :", df.columns, key="var_ligne")
    with col2:
        variable_colonne = st.selectbox("Variable pour les colonnes :", df.columns, key="var_colonne")

    type_pct = st.radio(
        "Type de pourcentage :",
        ["total", "ligne", "colonne"],
        format_func=lambda x: {
            "total": "🟦 Pourcentages totaux (pij = nij/n.. × 100)",
            "ligne": "🟩 Pourcentages ligne (pij = nij/ni. × 100)", 
            "colonne": "🟨 Pourcentages colonne (pij = nij/n.j × 100)"
        }[x],
        horizontal=True
    )

    if st.button("🔄 Générer le tableau corrigé", type="primary"):
        with st.spinner("Calcul en cours..."):
            tab = gen.generer_tableau_contingence_corrige(df, variable_ligne, variable_colonne, type_pct)
        
        st.success("✅ Tableau généré avec les formules statistiques correctes !")
        st.dataframe(tab, use_container_width=True)

        # Téléchargement
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            tab.to_excel(writer, index=True)
        buffer.seek(0)

        st.download_button(
            "📥 Télécharger le tableau Excel",
            data=buffer,
            file_name=f"tableau_contingence_{variable_ligne}_{variable_colonne}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# ======================================================================
# EXEMPLE D'UTILISATION
# ======================================================================

if __name__ == "__main__":
    # Test des fonctions
    generateur = DataGenerator()
    
    print("🧪 Génération d'un dataset de test...")
    df_test = generateur.generate_complex_dataset(100)
    print(f"Dataset généré : {df_test.shape}")
    
    print("\n📊 Test des tableaux de contingence corrigés :")
    tableau_test = generateur.generer_tableau_contingence_corrige(
        df_test, 'Type_Etablissement', 'Niveau_Complexite', 'ligne'
    )
    print(tableau_test)
    
    print("\n" + "="*60)
    print(generateur.afficher_formules_statistiques())