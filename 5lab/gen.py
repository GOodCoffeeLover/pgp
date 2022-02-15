test_size = 13500
file_name = f'test_{test_size}.txt'

with open(file_name, "wb") as file:
  file.write(int(test_size).to_bytes(4, 'little'))
  for i in range(test_size, -1, -1):
     file.write(int(i).to_bytes(4, 'little'))
 
