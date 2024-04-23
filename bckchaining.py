class Rule:
    def __init__(self, consequent, antecedent):
        self.consequent = consequent
        self.antecedent = antecedent

    def is_triggered(self, facts):
        return all(fact in facts for fact in self.antecedent)

class KnowledgeBase:
    def __init__(self):
        self.rules = []

    def add_rule(self, consequent, antecedent):
        rule = Rule(consequent, antecedent)
        self.rules.append(rule)

    def backward_chaining(self, target, facts):
        if target in facts:
            return True
        for rule in self.rules:
            if rule.consequent == target:
                if all(self.backward_chaining(antecedent, facts) for antecedent in rule.antecedent):
                    return True
        return False


kb = KnowledgeBase()

num_rules = int(input("Enter the number of rules: "))
for i in range(num_rules):
    consequent = input(f"Enter the consequent of rule {i + 1}: ")
    antecedent = input(f"Enter the antecedent of rule {i + 1} (separated by spaces): ").split()
    kb.add_rule(consequent, antecedent)

initial_facts = input("Enter the initial facts (separated by spaces): ").split()
target = input("Enter the target fact to check: ")


if kb.backward_chaining(target, initial_facts):
    print(f"The target '{target}' can be derived from the initial facts.")
else:
    print(f"The target '{target}' cannot be derived from the initial facts.")

#    sample Output:
# Enter the number of rules:  3
# Enter the consequent of rule 1:  C
# Enter the antecedent of rule 1 (separated by spaces):  A B
# Enter the consequent of rule 2:  D
# Enter the antecedent of rule 2 (separated by spaces):  B C
# Enter the consequent of rule 3:  E
# Enter the antecedent of rule 3 (separated by spaces):  D
# Enter the initial facts (separated by spaces):  A B
# Enter the target fact to check:  E
# The target 'E' can be derived from the initial facts.

