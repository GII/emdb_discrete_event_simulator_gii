perceptions = {}
sensors = ['cylinders', 'boxes']


for sensor in sensors:
    perceptions[sensor] = [{}]
    perceptions[sensor][0]['angle'] = 0

print(perceptions)