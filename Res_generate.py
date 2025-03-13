import matplotlib
from matplotlib import pyplot as plt


res1 = [7.0689, 10.423, 13.60, 15.39]
res2 = [6.5, 8.6, 11, 14]
res3 = [5.5, 7, 9.2, 12.3]

plt.figure(dpi=200)
plt.grid(linestyle="-.")
plt.plot([2, 4, 8, 16], res1, c="darkblue", linewidth=1, marker="o")
plt.plot([2, 4, 8, 16], res2, c="red", linewidth=1, marker="+")
plt.plot([2, 4, 8, 16], res3, c="purple", linewidth=1,ls='--', marker="s",markerfacecolor="white")
plt.xlabel("Num of AP antennas M")
plt.ylabel("Sum-SE(bits/Hz)")

plt.legend(["STAR-RIS assisted MADDPG", "RIS assisted MADDPG", "RIS assisted AO based"], loc=0)
plt.show()

# res1 = [5, 8.38, 9.68, 10.816, 11.60]
# res2 = [4.8, 5.9, 7.5, 8.9, 10]

# plt.figure(dpi=200)
# plt.grid(linestyle="-.")
# plt.plot([0, 1, 2, 3, 4], res1, c="darkblue", linewidth=1, marker="o")
# # plt.plot([2, 4, 8, 16], res2, c="red", linewidth=1, marker="+")
# plt.plot([0, 1, 2, 3, 4], res2, c="purple", linewidth=1, ls="--", marker="s", markerfacecolor="white")
# plt.xlabel("Num of RISs R")
# plt.ylabel("Sum-SE(bits/Hz)")

# plt.legend(["STAR-RIS assisted MADDPG", "RIS assisted MADDPG"], loc=0)
# plt.show()


# res1 = [8.5, 10.815, 11.05, 11.60, 12.1, 13.1328]
# res2 = [8.5, 9.3, 9.9, 10.2, 10.6, 11.6]

# plt.figure(dpi=200)
# plt.grid(linestyle="-.")
# plt.plot([0, 8, 12, 16, 32, 64], res1, c="darkblue", linewidth=1, marker="o")
# # plt.plot([2, 4, 8, 16], res2, c="red", linewidth=1, marker="+")
# plt.plot([0, 8, 12, 16, 32, 64], res2, c="purple", linewidth=1, ls="--", marker="s", markerfacecolor="white")
# plt.xlabel("Num of RIS elements N")
# plt.ylabel("Sum-SE(bits/Hz)")
# plt.xlim([-0.5,66])
# plt.legend(["STAR-RIS assisted MADDPG", "RIS assisted MADDPG"], loc=0)
# plt.show()
