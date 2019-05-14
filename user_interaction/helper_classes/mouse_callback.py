import cv2


def mouse_callback(event, x, y, _, param):
    """
    Responses on mouse callback

    :param param: Instance of object where click or move should be delegated
    :param event: event fired
    :param x: x position
    :param y: y position
    :param _: unused
    :param param: passed object
    """

    corridor_maker = param

    if event == cv2.EVENT_LBUTTONDOWN:
        corridor_maker.click(point=(x, y))

    corridor_maker.move(point=(x, y))
