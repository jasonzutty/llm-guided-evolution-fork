import ast
import math
import numpy as np
from collections import Counter

# =============================================================================
# Helper Functions for AST-based Comparison
# =============================================================================

def GetAstNodeCounts(code: str) -> dict:
    """
    Parse the code into an AST and return a dictionary counting the occurrences
    of each node type.
    
    Using AST ensures that we compare the structure of the code rather than mere text,
    making the diversity metric robust to cosmetic changes (e.g., comments, whitespace).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # In case of a syntax error, return an empty counter.
        return Counter()
    
    # Use a visitor to count node types
    counts = Counter()
    for node in ast.walk(tree):
        counts[type(node).__name__] += 1
    return counts

def ComputeCosineSimilarity(vectorA: dict, vectorB: dict) -> float:
    """
    Compute cosine similarity between two dictionaries representing feature vectors.
    The keys are the AST node types and values are their counts.
    """
    # Create a set of all keys from both vectors
    allKeys = set(vectorA.keys()).union(vectorB.keys())
    dotProduct = sum(vectorA.get(key, 0) * vectorB.get(key, 0) for key in allKeys)
    normA = math.sqrt(sum((vectorA.get(key, 0)) ** 2 for key in allKeys))
    normB = math.sqrt(sum((vectorB.get(key, 0)) ** 2 for key in allKeys))
    if normA == 0 or normB == 0:
        return 0.0
    return dotProduct / (normA * normB)

def ComputeAstDistance(code1: str, code2: str) -> float:
    """
    Compute a distance between two code strings based on their AST node counts.
    We first compute the cosine similarity between the AST feature vectors, and then
    define distance as 1 minus the cosine similarity.
    """
    counts1 = GetAstNodeCounts(code1)
    counts2 = GetAstNodeCounts(code2)
    similarity = ComputeCosineSimilarity(counts1, counts2)
    return 1 - similarity

# =============================================================================
# Diversity Quantifier Functions (using AST)
# =============================================================================

def ComputeIntraDiversity(population: list) -> float:
    """
    Compute the average pairwise AST distance among all individuals (code files)
    within a single island.
    
    This measures how structurally diverse the code files are.
    """
    n = len(population)
    if n < 2:
        return 0.0
    totalDistance = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            distance = ComputeAstDistance(population[i], population[j])
            totalDistance += distance
            count += 1
    return totalDistance / count

def ComputeDiversityContribution(individual: str, population: list) -> float:
    """
    Compute the diversity contribution of a single individual.
    This is the average AST distance between this individual and every other in the island.
    """
    if len(population) < 2:
        return 0.0
    distances = [ComputeAstDistance(individual, other) for other in population if other != individual]
    return sum(distances) / len(distances)

def ComputeInterDiversity(island1: list, island2: list) -> float:
    """
    Compute the average AST distance between every individual in island1 and every
    individual in island2.
    
    This provides a measure of how structurally different two islands are.
    """
    distances = []
    for code1 in island1:
        for code2 in island2:
            distances.append(ComputeAstDistance(code1, code2))
    if not distances:
        return 0.0
    return sum(distances) / len(distances)

def MigrationDecision(population: list, fitnessScores: list, alpha: float = 0.8, beta: float = 0.2, migrationRate: float = 0.2) -> list:
    """
    Decide which individuals should migrate based on a composite migration score.
    
    Each individual's migration score is a weighted combination of its fitness and its
    diversity contribution (structural uniqueness).
    
    Arguments:
        population: list of code strings.
        fitnessScores: list of fitness scores corresponding to each individual.
        alpha: weight for the fitness component.
        beta: weight for the diversity component.
        migrationRate: fraction of the population to select for migration.
    
    Returns:
        A list of code strings representing the migration candidates.
    """
    migrationScores = []
    for individual, fitness in zip(population, fitnessScores):
        diversityContribution = ComputeDiversityContribution(individual, population)
        # Combine fitness and diversity using the weighted sum
        migrationScore = alpha * fitness + beta * diversityContribution
        migrationScores.append(migrationScore)
    
    # Determine the number of individuals to migrate based on the migrationRate
    numToMigrate = max(1, int(len(population) * migrationRate))
    # Get the indices of the individuals with the highest migration scores
    indices = np.argsort(migrationScores)[-numToMigrate:]
    return [population[i] for i in indices]

# =============================================================================
# Example Usage & Testing
# =============================================================================

if __name__ == "__main__":
    # Example island population: individuals are represented as code strings.
    # These examples show slight structural differences.
    islandPopulation = [
        "def Foo():\n    return 1",
        "def Foo():\n    x = 2\n    return x",
        "def Foo():\n    for i in range(3):\n        print(i)\n    return 3",
        "def Foo():\n    if True:\n        return 4\n    else:\n        return 0"
    ]
    
    # Example fitness scores for these individuals (e.g., performance metrics)
    fitnessScores = [0.9, 0.8, 0.85, 0.95]
    
    # Compute the intra-island diversity based on AST differences
    intraDiversity = ComputeIntraDiversity(islandPopulation)
    print("Intra-Island Diversity (AST-based):", intraDiversity)
    
    # Decide which individuals should migrate based on the composite migration score
    migrationCandidates = MigrationDecision(islandPopulation, fitnessScores, alpha=0.7, beta=0.3, migrationRate=0.5)
    print("\nMigration Candidates (AST-based):")
    for candidate in migrationCandidates:
        print(candidate)
    
    # For demonstration, define a second island with different code structures.
    islandPopulation2 = [
        "def Bar():\n    return 'a'",
        "def Bar():\n    x = 'b'\n    return x",
        "def Bar():\n    for j in range(2):\n        print(j)\n    return 'c'"
    ]
    
    # Compute the inter-island diversity between the two islands
    interDiversity = ComputeInterDiversity(islandPopulation, islandPopulation2)
    print("\nInter-Island Diversity (AST-based):", interDiversity)