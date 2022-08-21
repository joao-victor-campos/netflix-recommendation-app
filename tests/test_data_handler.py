df3 = pd.DataFrame(
    [["c", 3, 10, "cat"], ["d", 4, 50, "dog"]],
    columns=["letter", "number", "number2", "animal"],
)


x = DataHandler(df3)
x.normalize(["number", "number2"])
print(x.one_hot_encode(["letter", "animal"]))
