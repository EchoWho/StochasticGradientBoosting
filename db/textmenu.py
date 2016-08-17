# Presents a numbered text menu for a series of items or from a GLOB string
#
# Output looks like the below where header = Select a dataset:
# ----- Select a dataset:  -----
#
# 0: Exit
# 1: helicopter_data
# 2: mg_10
# 3: normalized_take_off_data
# 4: projected_mocap_data
#
# Select a menu item: 1
#
# Arun Venkatraman (09-24-2015)
#

HEADER_LEADER = '----- '
HEADER_TRAILER = ' -----'


def textmenu(items_or_globstr, header="MENU", prompt="Select a menu item:"):
    print("\n" + HEADER_LEADER + header + HEADER_TRAILER + "\n")
    # convert to list if just a string
    if isinstance(items_or_globstr, str):
        import glob
        items = glob.glob(items_or_globstr)
    else:
        items = items_or_globstr

    for i, item in enumerate(items):
        print("{0:d}: {1:s}".format(i, str(item)))
    print("\n")

    choice = raw_input(prompt + " ")

    try:
        choice = int(choice)
    except:
        print("Invalid selection.")
        choice = None

    if choice < 0 or choice >= len(items):
        print('Selection out of range. Try again.')
        choice = None

    return choice
