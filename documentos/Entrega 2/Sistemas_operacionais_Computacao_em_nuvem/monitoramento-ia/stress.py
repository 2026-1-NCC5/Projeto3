import multiprocessing

def estressar_cpu():
    while True:
        x = 0
        for i in range(10000000):
            x += i * i

if __name__ == "__main__":

    processos = []

    for _ in range(multiprocessing.cpu_count()):

        p = multiprocessing.Process(target=estressar_cpu)

        p.start()

        processos.append(p)

    for p in processos:
        p.join()