def main():
    print("Hello world")
    addition(addition(1,2),5)
def addition(a,b):
    print(a+b)
    return a+b
if __name__ == "__main__":
    main()