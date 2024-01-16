def find_Ekman_layer_height(data_path, vis_path, file_name, u_G, steady_state_coord=None, make_plot=True):

    full_file_path = data_path + file_name
    # Open output file and load variables
    with h5py.File(full_file_path, "r+") as file:
        # perform byteswap to make handling with pandas dataframe possible
        u = file["u"][:]
        z = file["z"][:]
        t = file["t"][:]

    data = pd.DataFrame(data=u.T, columns=z.flatten())

    ekman_height_idx = np.zeros((1, len(t.flatten())))
    ekman_height_idx[...] = np.nan
    for row_idx in data.index:
        ekman_height_idx[0, row_idx] = np.argmax(np.around(data.iloc[row_idx, :], 1) == u_G)

    if make_plot:
        # Create mesh
        X, Y = np.meshgrid(t, z)

        # Make plot
        plt.figure(figsize=(5, 5))
        plt.pcolor(X, Y, u, cmap=cram.davos_r)
        plt.plot(
            t.flatten(),
            z[list(map(int, ekman_height_idx.flatten())), :],
            color="red",
            linestyle="--",
            label="Ekman layer",
        )

        cbar = plt.colorbar()
        cbar.set_label("u", rotation=0, labelpad=2)

        if steady_state_coord:
            plt.scatter(
                t.flatten()[int(steady_state_coord[0])],
                steady_state_coord[1],
                c="black",
                s=8,
                label="steady state",
                marker="x",
            )
        plt.ylim((0, 50))
        # plt.title(r"$u_G = $" + str(u_G))
        plt.xlabel("time [h]")
        plt.ylabel("z [m]")
        leg = plt.legend()
        for text in leg.get_texts():
            text.set_color("white")

        # Save plot
        if steady_state_coord:
            plt.savefig(
                vis_path
                + "/Ekman_layer_"
                + str(u_G)
                + "_h"
                + str(int(np.around(steady_state_coord[1])))
                + ".png",
                bbox_inches="tight",
                dpi=300,
            )
        else:
            plt.savefig(
                vis_path + "/Ekman_layer_" + str(u_G) + ".png",
                bbox_inches="tight",
                dpi=300,
            )


        # Clear memory
        plt.cla()  # Clear the current axes.
        plt.clf()  # Clear the current figure.
        plt.close("all")  # Closes all the figure windows.

    return z[list(map(int, ekman_height_idx.flatten())), :].flatten()