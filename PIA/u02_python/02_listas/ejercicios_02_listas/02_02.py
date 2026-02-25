beatles=[]

beatles.append("John Lennon")
beatles.append("Paul McCartney")
beatles.append("George Harrison")


beatles_update =["Stuart Sutcliffe","Pete Best"]

for i in range(len(beatles_update)):
    beatles.append(beatles_update[i])

del beatles[3:]

beatles.insert(0,"Ringo Starr")

print(beatles)
print(f"Los integrantes de los Beatles son: {len(beatles)}")
