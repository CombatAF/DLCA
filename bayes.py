# Step 1: run this command in jupyter notebook
# !pip install pgmpy --user and for normal terminal :
# pip install pgmpy --user

# Step 2:
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


model = BayesianNetwork([('A', 'C'), ('B', 'C'), ('C', 'D')])

# Define Conditional Probability Distributions (CPDs)
cpd_A = TabularCPD(variable='A', variable_card=2, values=[[0.8], [0.2]])
cpd_B = TabularCPD(variable='B', variable_card=2, values=[[0.7], [0.3]])
cpd_C = TabularCPD(variable='C', variable_card=2,
                   values=[[0.9, 0.6, 0.7, 0.1],
                           [0.1, 0.4, 0.3, 0.9]],
                   evidence=['A', 'B'], evidence_card=[2, 2])
cpd_D = TabularCPD(variable='D', variable_card=2,
                   values=[[0.95, 0.8], [0.05, 0.2]],
                   evidence=['C'], evidence_card=[2])


model.add_cpds(cpd_A, cpd_B, cpd_C, cpd_D)


model.check_model()


evidence_A = int(input("Enter evidence for node A (0 or 1): "))
evidence_B = int(input("Enter evidence for node B (0 or 1): "))

inference = VariableElimination(model)
query_result = inference.query(variables=['D'], evidence={'A': evidence_A, 'B': evidence_B})


print(query_result)
