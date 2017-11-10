# import arrow
# import quandl
#
#
# # num_days_back = 100
# # stock_ticker = "AAL"
# #
# #
# # print("GETTING STOCK DATA")
# #
# # end_date = arrow.now().format("YYYY-MM-DD")
# # start_date = arrow.now()
# # start_date = start_date.replace(days=(num_days_back * -1)).format("YYYY-MM-DD")
# #
# # quandl_api_key = "DqEaArDZQP8SfgHTd_Ko"
# # quandl.ApiConfig.api_key = quandl_api_key
# #
# # source = "WIKI/" + stock_ticker
# #
# # data = quandl.get(source, start_date=str(start_date), end_date=str(end_date))
# #
# # print(len(data))
# #
# # data = data[["Open", "High", "Low", "Volume", "Close"]].as_matrix()
#
# def solve(arr):
#
#
#     def recurse(y, x, arr, dparr, path, score):
#
#         new_score = score + arr[y][x]
#
#         if dparr[y][x] != None and new_score >= dparr[y][x]:
#             return None, None
#
#         dparr[y][x] = new_score
#
#
#
#         on_bottom = (y == len(arr) - 1)
#         on_right = (x == (len(arr[0]) - 1))
#
#         if on_bottom and on_right:
#             return path, new_score
#
#         elif on_bottom:
#             path += "r"
#             path, _ = recurse(y, x+1, arr, dparr, path, new_score)
#             return path, new_score
#
#         elif on_right:
#             path += "d"
#             path, _ = recurse(y+1, x, arr, dparr, path, new_score)
#             return path, new_score
#
#         else:
#             path1 = path + "r"
#             path1, score1 = recurse(y, x+1, arr, dparr, path1, new_score)
#
#             path2 = path + "d"
#             path2, score2 = recurse(y+1, x, arr, dparr, path2, new_score)
#
#             if score1 != None and score2 > score1:
#                 return path1, new_score
#             else:
#                 return path2, new_score
#
#
#     dparr = []
#
#     for minilist in arr:
#         dparr.append([None] * len(minilist))
#
#     path = ""
#     path, score = recurse(0, 0, arr, dparr, path, 0)
#     return path
#
#
# arr = [[4,3,4], [1,2,1], [0, 9, 8], [1, 2, -5], [4, 3, 7], [8, 4, 3], [5, 2, 8], [9, 1, 1], [4, 7, 1]]
#
# print(solve(arr))

ex = "one"

print(ex.split(" "))
