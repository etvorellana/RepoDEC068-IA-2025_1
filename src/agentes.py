

class Agente:
    
    def __init__(self):
        self.percepts = []
    
    def sensor(self, ambiente):
        pass

    def atuador(self, ambiente, action):
        pass

    def agentProgram(self, percept):
        pass


class MundoAspirador:

    def __init__(self, loc = 'A', ambiente = None):
        if ambiente is None:
            self.ambiente = {'A':'Sujo', 'B':'Sujo'}
        else:
            self.ambiente = ambiente
        self.loc = loc
    
    def percept(self):
        return (self.loc, self.ambiente[self.loc])
    
    def execute_action(self, action):
        if action == 'Aspirar':
            self.ambiente[self.loc] = ('Limpo')
        elif action == 'Right':
            self.loc = 'B'
        elif action == 'Left':
            self.loc = 'A'
    
    def __str__(self):
        return str(self.ambiente)
    

class AgenteDirigidoPorTabela(Agente):
    
    def __init__(self, table):
        super().__init__()
        self.table = table

    def sensor(self, ambiente):
        return ambiente.percept()
        
    def atuador(self, ambiente, action):
        ambiente.execute_action(action)
        return ambiente

    def agentProgram(self, percept):
        self.percepts.append(percept)
        if self.percepts[-1] in self.table:
            return self.table[self.percepts[-1]]
        else:
            return 'None'

 
class AgenteReativo(Agente):
        
    def __init__(self):
        super().__init__()
        
    def sensor(self, ambiente):
        return ambiente.percept()
        
    def atuador(self, ambiente, action):
        ambiente.execute_action(action)
        return ambiente
    
    def agentProgram(self, percept):
        location, status = percept
        if status == 'Sujo':
            return 'Aspirar'
        elif location == 'A':
            return 'Right'
        elif location == 'B':
            return 'Left'       

def main():
    # Testando o agente
    # Ambiente: A e B sujos

    print('--- Teste do Agente Dirigido por Tabela ---')
    ambiente = MundoAspirador('A', {'A':'Sujo', 'B':'Sujo'})
    table = {
        ('A', 'Limpo'): 'Right',
        ('B', 'Limpo'): 'Left',
        ('A', 'Sujo'): 'Aspirar',
        ('B', 'Sujo'): 'Aspirar'
    }
    agente = AgenteDirigidoPorTabela(table)
    for step in range(5):
        print('Step: ', step, end=' ')
        percept = agente.sensor(ambiente)
        action = agente.agentProgram(percept)
        print('Percept: ', percept)
        print('Action: ', action)
        ambiente = agente.atuador(ambiente, action)
        print('Ambiente: ', ambiente)
        print('---')

    print('--- Teste do Agente Reativo ---')
    ambiente = MundoAspirador('A', {'A':'Sujo', 'B':'Sujo'})
    agente = AgenteReativo()
    for step in range(5):
        print('Step: ', step, end=' ')
        percept = agente.sensor(ambiente)
        action = agente.agentProgram(percept)
        print('Percept: ', percept)
        print('Action: ', action)
        ambiente = agente.atuador(ambiente, action)
        print('Ambiente: ', ambiente)

if __name__ == "__main__":
    main()