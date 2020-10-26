from related_work.AutoCross.dataset.dataset import amazon

if __name__ == '__main__':
    train, label = amazon()
    l = train.columns
    print(l)
    l.append('a')
    print(l)