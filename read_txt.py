my_file = open("ms_fs.txt", "r")
content = my_file.read()
print(content)

content_list = content.split(",")
my_file.close()
print(content_list)
