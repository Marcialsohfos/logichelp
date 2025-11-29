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
            'Type_Etablissement': ['Public', 'Privé', 'Confessionnel'],
            'Niveau_Complexite': ['Level I', 'Level II', 'Level III', 'Level IV'],
            'Specialite': ['Généraliste', 'Cardiologie', 'Pédiatrie', 'Chirurgie', 'Urgence'],
            'Statut': ['Public', 'Privé', 'Mixte'],
            'Zone': ['Urbaine', 'Rurale', 'Périurbaine'],
            'Accreditation': ['Oui', 'Non', 'En cours'],
            'Equipement': ['Basique', 'Intermédiaire', 'Avancé'],
            'Personnel': ['Insuffisant', 'Adéquat', 'Abondant'],
            'Financement': ['Etat', 'Privé', 'International', 'Mixte'],
            'Glycosurie_Albuminurie': ['Oui', 'Non']  # Ajout pour l'exemple
        }
        
        category_keys = list(categories.keys())
        
        for i in range(n_vars):
            if i < len(category_keys):
                var_name = category_keys[i]
                categories_list = categories[var_name]
            else:
                var_name = f"Cat_Var_{i+1}"
                categories_list = [f'Cat_{j}' for j in range(random.randint(3, 8))]
            
            # Générer des probabilités réalistes pour chaque catégorie
            if var_name == 'Type_Etablissement':
                probs = [0.70, 0.20, 0.10]  # Public, Privé, Confessionnel
            elif var_name == 'Niveau_Complexite':
                probs = [0.50, 0.35, 0.10, 0.05]  # Level I, II, III, IV
            elif var_name == 'Glycosurie_Albuminurie':
                probs = [0.30, 0.70]  # 30% Oui, 70% Non
            else:
                probs = [1/len(categories_list)] * len(categories_list)
            
            vars_dict[var_name] = np.random.choice(
                categories_list, 
                n_obs,
                p=probs
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
                
                # Arrondir selon le type de variable
                if 'Age' in var_name or 'Nombre' in var_name or 'Personnel' in var_name:
                    values = np.round(values).astype(int)
                else:
                    values = np.round(values, 2)
                
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
        
        for i, key in enumerate(numerical_keys[:3]):
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
    
    def generate_healthcare_dataset(self, n_observations=1000):
        """
        Génère un dataset spécifique au domaine de la santé
        """
        data = {}
        
        # Variables démographiques
        data['Age'] = np.random.normal(45, 15, n_observations).clip(18, 90).astype(int)
        data['Sexe'] = np.random.choice(['M', 'F'], n_observations, p=[0.48, 0.52])
        data['Region'] = np.random.choice(['Nord', 'Sud', 'Est', 'Ouest', 'Centre'], n_observations)
        
        # Variables des établissements
        data['Type_Etablissement'] = np.random.choice(
            ['Public', 'Privé', 'Confessionnel'], 
            n_observations, 
            p=[0.7, 0.2, 0.1]
        )
        data['Niveau_Complexite'] = np.random.choice(
            ['Level I', 'Level II', 'Level III', 'Level IV'], 
            n_observations, 
            p=[0.5, 0.35, 0.1, 0.05]
        )
        
        # Variables médicales
        data['Glycosurie_Albuminurie'] = np.random.choice(
            ['Oui', 'Non'], 
            n_observations, 
            p=[0.3, 0.7]
        )
        
        # Variables de performance
        data['Taux_Occupation'] = np.random.beta(2, 2, n_observations) * 0.65 + 0.3
        data['Taux_Occupation'] = np.round(data['Taux_Occupation'] * 100, 2)
        data['Patients_Jour'] = np.random.poisson(30, n_observations).clip(5, 100)
        data['Satisfaction'] = np.random.normal(7.5, 1.5, n_observations).clip(1, 10).round(1)
        
        # Variables financières
        data['Budget_Annuel'] = np.random.lognormal(12, 1.5, n_observations).clip(50000, 5000000).round(2)
        data['Cout_Moyen_Sejour'] = np.random.lognormal(6, 1, n_observations).clip(100, 5000).round(2)
        
        # Variables binaires
        data['Urgence_Disponible'] = np.random.choice([0, 1], n_observations, p=[0.3, 0.7])
        data['Laboratoire_Interne'] = np.random.choice([0, 1], n_observations, p=[0.4, 0.6])
        data['Radiologie'] = np.random.choice([0, 1], n_observations, p=[0.5, 0.5])
        
        df = pd.DataFrame(data)
        return df