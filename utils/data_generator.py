# ================================================================
# data_generator.py - Version FINALE CORRIGÉE
# ================================================================

import pandas as pd
import numpy as np
from faker import Faker
import random
import streamlit as st
import io

# ================================================================
#  CLASS: DATA GENERATOR
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
        """Génère des variables catégorielles réalistes"""
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
        """Génère des variables numériques réalistes avec différentes distributions"""
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
                else:
                    values = np.random.normal(0, 1, n_obs)

                values = np.clip(values, cfg["min"], cfg["max"])

            else:
                name = f"Num_Var_{i+1}"
                values = np.random.normal(0, 1, n_obs)
                values = np.round(values, 2)

            variables[name] = values

        return variables

    # ------------------------------------------------------------
    def _generate_binary_variables(self, n_vars, n_obs):
        """Génère des variables binaires (0/1)"""
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
        """Génère une variable cible corrélée avec d'autres variables"""
        target = np.random.normal(0, 1, n_obs)

        # Identifier les variables numériques pour créer des corrélations
        numeric_keys = [
            k
            for k, v in data.items()
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number)
        ]

        # Ajouter des corrélations avec les 3 premières variables numériques
        for key in numeric_keys[:3]:
            if len(data[key]) == n_obs and np.std(data[key]) > 0:
                target += 0.3 * (data[key] - np.mean(data[key])) / np.std(data[key])

        # Convertir en variable catégorielle
        q = np.percentile(target, [25, 50, 75])
        idx = np.digitize(target, q)

        labels = ["Faible", "Moyen", "Élevé", "Très élevé"]
        return [labels[min(i, 3)] for i in idx]

    # ------------------------------------------------------------
    def _add_missing_values(self, df, percentage):
        """Ajoute des valeurs manquantes aléatoires"""
        if percentage <= 0:
            return df
            
        df2 = df.copy()
        n_missing = int(df2.size * percentage / 100)

        for _ in range(n_missing):
            row = random.randint(0, df2.shape[0] - 1)
            col = random.choice(df2.columns)
            df2.loc[row, col] = np.nan

        return df2

# ============================================================
# TABLEAUX DE CONTINGENCE - VERSION FINALE CORRIGÉE
# ============================================================
def generer_tableau_contingence_corrige(self, df, var_ligne, var_col, mode="total"):
    """
    Génère un tableau de contingence avec formules statistiques correctes.
    Format: effectif (pourcentage%) pour les cellules internes, effectifs seulement pour les totaux
    """
    
    # Vérification des colonnes
    if var_ligne not in df.columns or var_col not in df.columns:
        raise ValueError(f"Variables non trouvées: {var_ligne} ou {var_col}")

    # Tableau d'effectifs avec marges
    effectifs = pd.crosstab(
        df[var_ligne], 
        df[var_col], 
        margins=True, 
        margins_name="Total"
    )
    
    n_total = effectifs.loc["Total", "Total"]

    # Tableau des pourcentages selon le mode
    pourcent = effectifs.copy().astype(float)

    for i in effectifs.index:
        for j in effectifs.columns:
            nij = effectifs.loc[i, j]

            # MODE TOTAL
            if mode == "total":
                # Pourcentages sur le total général pour toutes les cellules
                pourcent.loc[i, j] = 100 * nij / n_total if n_total > 0 else 0.0

            # MODE LIGNE
            elif mode == "ligne":
                if i == "Total":
                    # Dernière ligne : pourcentage sur total général
                    pourcent.loc[i, j] = 100 * nij / n_total if n_total > 0 else 0.0
                else:
                    # Cellules internes : pourcentage par ligne
                    denom = effectifs.loc[i, "Total"]
                    pourcent.loc[i, j] = 100 * nij / denom if denom > 0 else 0.0

            # MODE COLONNE
            elif mode == "colonne":
                if j == "Total":
                    # Dernière colonne : pourcentage sur total général
                    pourcent.loc[i, j] = 100 * nij / n_total if n_total > 0 else 0.0
                else:
                    # Cellules internes : pourcentage par colonne
                    denom = effectifs.loc["Total", j]
                    pourcent.loc[i, j] = 100 * nij / denom if denom > 0 else 0.0

    # Construction du tableau final
    final = effectifs.copy().astype(object)
    
    for i in effectifs.index:
        for j in effectifs.columns:
            e = effectifs.loc[i, j]
            p = round(float(pourcent.loc[i, j]), 1)

            # Cellules internes : effectif (pourcentage%)
            if i != "Total" and j != "Total":
                final.loc[i, j] = f"{e} ({p}%)"
            
            # Cellules des totaux : effectif seulement
            else:
                final.loc[i, j] = f"{e}"

    return final

    # ------------------------------------------------------------
    def afficher_formules_statistiques(self):
        """Retourne les formules statistiques utilisées"""
        return """
📊 **FORMULES STATISTIQUES APPLIQUÉES**

**Notations :**
- $n_{..}$ = effectif total  
- $n_{ij}$ = effectif de la cellule $(i,j)$  
- $n_{i.}$ = total de la ligne $i$  
- $n_{.j}$ = total de la colonne $j$  

**Types de pourcentages :**

🟦 **POURCENTAGES TOTAUX**
• Cellules : $p_{ij} = n_{ij} / n_{..} \times 100$
• Totaux : effectifs seulement.

🟩 **POURCENTAGES LIGNE**
• Cellules : $p_{ij} = n_{ij} / n_{i.} \times 100$
• **Total de Ligne (marge) :** Effectif suivi de **100.0** (sans le `%`) pour confirmer la somme des pourcentages de ligne.
• Autres Totaux : effectifs seulement.

🟨 **POURCENTAGES COLONNE**
• Cellules : $p_{ij} = n_{ij} / n_{.j} \times 100$
• **Total de Colonne (marge) :** Effectif suivi de **100.0** (sans le `%`) pour confirmer la somme des pourcentages de colonne.
• Autres Totaux : effectifs seulement.

**Particularités :**
• Le coin Total-Total affiche l'effectif général.
• Arrondi à 1 décimale pour tous les pourcentages des cellules internes.
"""


# ================================================================
#  INTERFACE STREAMLIT
# ================================================================
def creer_interface_tableaux_contingence(df):
    """
    Crée une interface Streamlit pour les tableaux de contingence
    """
    
    st.header("📊 Tableaux de Contingence - Version Finale Corrigée")

    gen = DataGenerator()
    
    # Section informations
    with st.expander("ℹ️ Informations et formules"):
        st.markdown(gen.afficher_formules_statistiques())
        st.info(
            "**Note importante :** En mode **Ligne** ou **Colonne**, le total de marge correspondant affiche **(100.0)** après l'effectif pour indiquer que la somme des pourcentages des cellules internes est bien 100%."
        )

    # Sélection des variables
    col1, col2 = st.columns(2)
    with col1:
        var_ligne = st.selectbox(
            "Variable pour les lignes :", 
            df.columns,
            help="Variable qui déterminera les lignes du tableau"
        )
    with col2:
        var_col = st.selectbox(
            "Variable pour les colonnes :", 
            df.columns,
            help="Variable qui déterminera les colonnes du tableau"
        )

    # Sélection du mode
    mode = st.radio(
        "Type de pourcentage :", 
        ["total", "ligne", "colonne"], 
        horizontal=True,
        format_func=lambda x: {
            "total": "🟦 Pourcentages totaux",
            "ligne": "🟩 Pourcentages ligne", 
            "colonne": "🟨 Pourcentages colonne"
        }[x]
    )

    # Bouton de génération
    if st.button("🔄 Générer le tableau", type="primary"):
        try:
            with st.spinner("Calcul du tableau en cours..."):
                tab = gen.generer_tableau_contingence_corrige(df, var_ligne, var_col, mode)
            
            st.success("✅ Tableau généré avec succès !")
            
            # Affichage du tableau
            st.dataframe(tab, use_container_width=True)
            
            # Téléchargement Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                tab.to_excel(writer, sheet_name="Tableau_Contingence", index=True)
            output.seek(0)

            st.download_button(
                "📥 Télécharger en Excel",
                data=output.getvalue(),
                file_name=f"tableau_contingence_{var_ligne}_{var_col}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la génération : {str(e)}")


# ================================================================
# TEST
# ================================================================
if __name__ == "__main__":
    # Test des fonctions
    print("🧪 Test du DataGenerator...")
    
    gen = DataGenerator()
    
    # Génération d'un dataset de test
    df = gen.generate_complex_dataset(300)
    print(f"✅ Dataset généré : {df.shape[0]} observations, {df.shape[1]} variables")
    
    # Test des tableaux de contingence (mode ligne)
    print("\n📋 Test tableau de contingence (mode ligne) :")
    tableau_test_ligne = gen.generer_tableau_contingence_corrige(
        df, "Type_Etablissement", "Niveau_Complexite", "ligne"
    )
    print(tableau_test_ligne)

    # Test des tableaux de contingence (mode colonne)
    print("\n📋 Test tableau de contingence (mode colonne) :")
    tableau_test_colonne = gen.generer_tableau_contingence_corrige(
        df, "Type_Etablissement", "Niveau_Complexite", "colonne"
    )
    print(tableau_test_colonne)