# ================================================================
# data_generator.py - Version nettoyée, restructurée et corrigée
# ================================================================

import pandas as pd
import numpy as np
from faker import Faker
import random

# ================================================================
#  CLASS: DATA GENERATOR
# ================================================================
class DataGenerator:
    """
    Générateur avancé de données fictives réalistes,
    incluant un module robuste de tableaux de contingence.
    """

    # ------------------------------------------------------------
    def __init__(self, seed=42):
        self.fake = Faker("fr_FR")
        np.random.seed(seed)
        random.seed(seed)

    # ------------------------------------------------------------
    def generate_complex_dataset(
        self,
        n_observations=1000,
        n_categorical=5,
        n_numerical=7,
        n_binary=3,
        missing_percentage=5.0,
    ):
        """Génère un dataset complexe complet."""
        data = {}

        data.update(self._generate_categorical_variables(n_categorical, n_observations))
        data.update(self._generate_numerical_variables(n_numerical, n_observations))
        data.update(self._generate_binary_variables(n_binary, n_observations))

        data["Var_Interet"] = self._generate_target_variable(data, n_observations)

        df = pd.DataFrame(data)

        if missing_percentage > 0:
            df = self._add_missing_values(df, missing_percentage)

        return df

    # ------------------------------------------------------------
    def _generate_categorical_variables(self, n_vars, n_obs):
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
            "Financement": ["Etat", "Privé", "International", "Mixte"],
        }

        variables = {}
        keys = list(predefined.keys())

        for i in range(n_vars):
            if i < len(keys):
                name = keys[i]
                categories = predefined[name]
            else:
                name = f"Cat_Var_{i+1}"
                categories = [f"Cat_{j}" for j in range(random.randint(3, 8))]

            variables[name] = np.random.choice(categories, size=n_obs)

        return variables

    # ------------------------------------------------------------
    def _generate_numerical_variables(self, n_vars, n_obs):
        configs = [
            {"name": "Age_Patients", "dist": "normal", "params": [45, 15], "min": 18, "max": 90},
            {"name": "Nombre_Lits", "dist": "poisson", "params": [50], "min": 10, "max": 200},
            {"name": "Budget_Annuel", "dist": "lognormal", "params": [12, 1.5], "min": 50000, "max": 5_000_000},
            {"name": "Personnel_Medical", "dist": "normal", "params": [25, 10], "min": 5, "max": 100},
            {"name": "Patients_Jour", "dist": "poisson", "params": [30], "min": 5, "max": 100},
            {"name": "Taux_Occupation", "dist": "beta", "params": [2, 2], "min": 0.3, "max": 0.95},
            {"name": "Distance_Hopital", "dist": "exponential", "params": [0.1], "min": 0, "max": 50},
            {"name": "Satisfaction_Patients", "dist": "normal", "params": [7.5, 1.5], "min": 1, "max": 10},
            {"name": "Duree_Sejour", "dist": "gamma", "params": [2, 2], "min": 1, "max": 30},
            {"name": "Cout_Operation", "dist": "lognormal", "params": [8, 1], "min": 100, "max": 10_000},
        ]

        variables = {}

        for i in range(n_vars):
            if i < len(configs):
                cfg = configs[i]
                name = cfg["name"]
                dist = cfg["dist"]

                if dist == "normal":
                    values = np.random.normal(*cfg["params"], n_obs)
                elif dist == "poisson":
                    values = np.random.poisson(cfg["params"][0], n_obs)
                elif dist == "lognormal":
                    values = np.random.lognormal(*cfg["params"], n_obs)
                elif dist == "beta":
                    values = np.random.beta(*cfg["params"], n_obs)
                elif dist == "exponential":
                    values = np.random.exponential(cfg["params"][0], n_obs)
                elif dist == "gamma":
                    values = np.random.gamma(*cfg["params"], n_obs)

                values = np.clip(values, cfg["min"], cfg["max"])

            else:
                name = f"Num_Var_{i+1}"
                values = np.random.normal(0, 1, n_obs)

            variables[name] = values

        return variables

    # ------------------------------------------------------------
    def _generate_binary_variables(self, n_vars, n_obs):
        configs = [
            {"name": "Urgence_Disponible", "p": 0.7},
            {"name": "Laboratoire_Interne", "p": 0.6},
            {"name": "Radiologie", "p": 0.5},
            {"name": "Pharmacy", "p": 0.8},
            {"name": "Ambulance", "p": 0.4},
            {"name": "Bloc_Operatoire", "p": 0.3},
            {"name": "Soins_Intensifs", "p": 0.2},
        ]

        variables = {}

        for i in range(n_vars):
            if i < len(configs):
                name = configs[i]["name"]
                p = configs[i]["p"]
            else:
                name = f"Bin_Var_{i+1}"
                p = random.uniform(0.2, 0.8)

            variables[name] = np.random.choice([0, 1], size=n_obs, p=[1 - p, p])

        return variables

    # ------------------------------------------------------------
    def _generate_target_variable(self, data, n_obs):
        target = np.random.normal(0, 1, n_obs)

        numeric_keys = [
            k
            for k, v in data.items()
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number)
        ]

        for key in numeric_keys[:3]:
            target += 0.3 * (data[key] - np.mean(data[key])) / np.std(data[key])

        q = np.percentile(target, [25, 50, 75])
        idx = np.digitize(target, q)

        labels = ["Faible", "Moyen", "Élevé", "Très élevé"]
        return [labels[min(i, 3)] for i in idx]

    # ------------------------------------------------------------
    def _add_missing_values(self, df, percentage):
        df2 = df.copy()
        n_missing = int(df2.size * percentage / 100)

        for _ in range(n_missing):
            row = random.randint(0, df2.shape[0] - 1)
            col = random.choice(df2.columns)
            df2.loc[row, col] = np.nan

        return df2

    # ============================================================
    # TABLEAUX DE CONTINGENCE — VERSION STATISTIQUE COMPLÈTE (Option B)
    # ============================================================
    def generer_tableau_contingence_corrige(self, df, var_ligne, var_col, mode="total"):
        """
        Option B : Totaux suivent les règles strictes :
        - total colonne = n.j / n.. × 100
        - total ligne   = ni. / n.. × 100
        - coin final = 100%
        """

        effectifs = pd.crosstab(df[var_ligne], df[var_col], margins=True, margins_name="Total")
        n_total = effectifs.loc["Total", "Total"]

        pourcent = effectifs.copy().astype(float)

        for i in effectifs.index:
            for j in effectifs.columns:
                nij = effectifs.loc[i, j]

                # Coin final
                if i == "Total" and j == "Total":
                    pourcent.loc[i, j] = 100
                    continue

                # -----------------------------
                # MODE TOTAL
                # -----------------------------
                if mode == "total":
                    pourcent.loc[i, j] = 100 * nij / n_total
                    continue

                # -----------------------------
                # MODE LIGNE
                # -----------------------------
                if mode == "ligne":
                    if j == "Total":
                        # total ligne = ni. / n..
                        pourcent.loc[i, j] = 100 * nij / n_total
                    elif i == "Total":
                        # total colonne = n.j / n..
                        pourcent.loc[i, j] = 100 * nij / n_total
                    else:
                        denom = effectifs.loc[i, "Total"]
                        pourcent.loc[i, j] = 100 * nij / denom if denom else 0
                    continue

                # -----------------------------
                # MODE COLONNE
                # -----------------------------
                if mode == "colonne":
                    if j == "Total":
                        # total ligne = ni. / n..
                        pourcent.loc[i, j] = 100 * nij / n_total
                    elif i == "Total":
                        # total colonne = n.j / n..
                        pourcent.loc[i, j] = 100 * nij / n_total
                    else:
                        denom = effectifs.loc["Total", j]
                        pourcent.loc[i, j] = 100 * nij / denom if denom else 0
                    continue

        # Combinaison Effectifs + Pourcentages
        final = effectifs.copy().astype(object)
        for i in effectifs.index:
            for j in effectifs.columns:
                e = effectifs.loc[i, j]
                p = round(float(pourcent.loc[i, j]), 1)
                final.loc[i, j] = f"{e} ({p}%)"

        return final

    # ------------------------------------------------------------
    def afficher_formules_statistiques(self):
        return """
📊 **FORMULES STATISTIQUES SUIVIES (Option B)**

n.. : total général  
nij : cellule (i,j)  
ni. : total de la ligne  
n.j : total de la colonne  

### ➤ Pourcentage TOTAL
pij = nij / n.. × 100  

### ➤ Pourcentage LIGNE
pij = nij / ni. × 100  
fi. = ni. / n.. × 100  
f.j = n.j / n.. × 100  

### ➤ Pourcentage COLONNE
pij = nij / n.j × 100  
fi. = ni. / n.. × 100  
f.j = n.j / n.. × 100  

Coin final = 100%
"""


# ================================================================
#  STREAMLIT (OPTIONNEL)
# ================================================================
def creer_interface_tableaux_contingence(df):
    import streamlit as st
    import io

    st.header("📊 Tableaux de Contingence — Version corrigée")

    gen = DataGenerator()
    with st.expander("📘 Voir les formules"):
        st.markdown(gen.afficher_formules_statistiques())

    col1, col2 = st.columns(2)
    var_ligne = col1.selectbox("Variable ligne :", df.columns)
    var_col = col2.selectbox("Variable colonne :", df.columns)

    mode = st.radio("Type de pourcentage :", ["total", "ligne", "colonne"], horizontal=True)

    if st.button("Générer"):
        tab = gen.generer_tableau_contingence_corrige(df, var_ligne, var_col, mode)
        st.dataframe(tab)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            tab.to_excel(writer, index=True)

        st.download_button("Télécharger Excel", output.getvalue(), f"tableau_{var_ligne}_{var_col}.xlsx")


# ================================================================
# TEST LOCAL
# ================================================================
if __name__ == "__main__":
    gen = DataGenerator()
    df = gen.generate_complex_dataset(300)
    print(gen.generer_tableau_contingence_corrige(df, "Type_Etablissement", "Niveau_Complexite", "ligne"))
