from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import cv2, numpy as np, pygame
from solver.board import Board
from sounds.play_audio import playsound

# Global variables
dragging_kp = None
original_image = None
image = None

DARK_BROWN = (40, 10, 15)
BURN = (65, 22, 30)
DARK_GREEN = (0, 160, 0)
GREEN = (0, 255, 0)
CREAM = (230, 223, 215)
BROWN = (108, 35, 44)
LIGHT_BLUE = (136, 148, 183)
DARK_BLUE = (43, 59, 112)
MAGENTA = (255, 0, 255)
GOLD = (180, 150, 40)
RED = (208, 46, 85)
TAN = (142, 82, 71)
TILE_SIZE = 40
GAP_SIZE = 2

def mouse_callback(event, x, y, flags, params):
    global dragging_kp, original_image, image
    keypoints = params

    if event == cv2.EVENT_LBUTTONDOWN:
        for i, kp in enumerate(keypoints):
            if np.sqrt((x - kp[0])**2 + (y - kp[1])**2) < 10:  # 10 is the radius of the circle
                dragging_kp = i
                break

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_kp is not None:
            keypoints[dragging_kp] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging_kp = None

    image = original_image.copy()  # Create a fresh copy of the original image
    for i, kp in enumerate(keypoints):
        dest_pt = keypoints[(i + 1) % len(keypoints)]
        image = cv2.line(image, tuple(kp), tuple(dest_pt), (0,255,0), 2)
        image = cv2.circle(image, tuple(kp), 1, (255,0,255), 4)
        if i == 3:  # redraw first dot to make sure it is visible
            image = cv2.circle(image, tuple(keypoints[0]), 1, (255,0,255), 4)

def visualize_keypoints(input_image, keypoints):
    print("Found the corners! Let's fucking go!")
    print("Click and drag the magenta points if their positions look off. Press 'q' when you're done.")
    
    global original_image, image
    original_image = input_image
    image = original_image.copy()

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback, keypoints)

    while True:
        cv2.imshow('image', image)
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

def put_tile(x, y, row, col, board, screen):
    char = board.board[row][col]
    letter_mult = board.LETTER_MULTIPLIERS[(row, col)]
    word_mult = board.WORD_MULTIPLIERS[(row, col)]

    if word_mult == 3:
        draw_tile(screen, x, y, char, board.VALUES, bg_color=BROWN)
    elif word_mult == 2:
        draw_tile(screen, x, y, char, board.VALUES, bg_color=RED)
    elif letter_mult == 3:
        draw_tile(screen, x, y, char, board.VALUES, bg_color=DARK_BLUE)
    elif letter_mult == 2:
        draw_tile(screen, x, y, char, board.VALUES, bg_color=LIGHT_BLUE)
    else:
        draw_tile(screen, x, y, char, board.VALUES)

# Function to draw a tile with background and text (if any)
def draw_tile(screen, x, y, text, values, bg_color=CREAM, text_color=CREAM):
    bkgd = TAN if text and bg_color != DARK_GREEN else bg_color  # green text for illustrating plays
    pygame.draw.rect(screen, bkgd, (x, y, TILE_SIZE, TILE_SIZE))
    
    if text:
        char_font = pygame.font.Font(None, 32)
        pts_font = pygame.font.Font(None, 15)
        text_surface = char_font.render(text, True, text_color)
        pts_surface = pts_font.render(str(values[text]), True, text_color)

        text_rect = text_surface.get_rect(center=(x - 2 + TILE_SIZE // 2, y + TILE_SIZE // 2))
        pts_rect = pts_surface.get_rect(center=(x - 9 + TILE_SIZE, y - 9 + TILE_SIZE))
        screen.blit(text_surface, text_rect)
        screen.blit(pts_surface, pts_rect)

def display_board(board, fails, rambling):
    class TextInputBox:
        def __init__(self, x, y, width, height, font):
            self.x = x
            self.y = y
            self.width = width
            self.height = height
            self.font = font
            self.text = ""
            self.active = False  # Flag to track if the box is active for input

        def draw(self, screen):
            if self.active:
                # Draw background rect
                color = MAGENTA
                pygame.draw.rect(screen, color, (self.x, self.y, self.width, self.height))

                # Render and draw text
                text_surface = self.font.render(self.text, True, (0, 0, 0))
                text_rect = text_surface.get_rect(center=(self.x - 2 + self.width // 2, self.y + self.height // 2))
                screen.blit(text_surface, text_rect)

            elif self.text.isalpha() and not self.active:
                draw_tile(screen, self.x, self.y, self.text.lower(), board.VALUES)
                row = (self.y - GAP_SIZE) // (GAP_SIZE + TILE_SIZE)
                col = (self.x - GAP_SIZE) // (GAP_SIZE + TILE_SIZE)
                board.board[row][col] = self.text.lower()
                
            elif not self.active:
                draw_tile(screen, self.x, self.y, '?', board.VALUES)

        def handle_keydown(self, event):
            if self.active:
                if event.key == pygame.K_RETURN:  # Enter key pressed - validate and reset
                    if self.text.isalpha():  # Check if only letters
                        # Process valid input (replace with your desired action)
                        self.active = False
                    else:
                        playsound('sounds/bruh.wav')
                else:
                    self.text = event.unicode

        def handle_click(self, event):
            if self.x <= event.pos[0] <= self.x + self.width and self.y <= event.pos[1] <= self.y + self.height:
                self.active = True
            else:
                self.active = False  # Deactivate if clicked outside

    text_boxes = []
    screen_width = 15 * TILE_SIZE + 16 * GAP_SIZE
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_width))
    pygame.display.set_caption("Scrabble Board")

    screen.fill(GOLD)
    for row in range(15):
        for col in range(15):
            x = col * TILE_SIZE + (col + 1) * GAP_SIZE
            y = row * TILE_SIZE + (row + 1) * GAP_SIZE
            char = board.board[row][col]
            if char == '?':
                text_boxes.append(TextInputBox(x, y, TILE_SIZE, TILE_SIZE, pygame.font.Font(None, 32)))
            put_tile(x, y, row, col, board, screen)

    if text_boxes:
        if fails > 0:
            playsound("sounds/bruh.wav")
        else:
            playsound("sounds/reassign_blanks.wav")
        print('Click on all question marks and change them to intended letters. Press enter to lock in a change.')
    else:
        playsound(f"sounds/morty{rambling % 2 + 1}.wav")
        print('Click on any inaccurate tiles to correct them. Press enter to lock in a change.')

    # Game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if not text_boxes:
                    playsound('sounds/morty3.wav')
                else:
                    for box in text_boxes:
                        box.handle_click(event)
                        box.draw(screen)
            if event.type == pygame.KEYDOWN:
                for box in text_boxes:
                    box.handle_keydown(event)
                    box.draw(screen)

        pygame.display.flip()

    pygame.display.quit()
    for row in range(15):
        for col in range(15):
            if board.board[row][col] == '?':
                return True
            
    return False

def display_plays(plays, board):
    class ScrollableList:
        def __init__(self, x, y, width, height, font=None, item_height=35):
            self.x = x
            self.y = y
            self.width = width
            self.height = height
            self.pts_font = font  # Font for text items (if applicable)
            self.word_font = pygame.font.Font(None, 28)
            self.selected = plays[0]
            self.prev = plays[1]
            self.item_height = item_height  # Height of each list item
            self.scroll_pos = 0  # Current scroll position (index of first visible item)

        def get_visible_items(self):
            # Calculate visible items based on scroll position and viewport size
            start_index = max(0, self.scroll_pos)  # Clamp to avoid negative index
            end_index = min(len(plays), start_index + int(self.height / self.item_height) + 1)
            return plays[start_index:end_index]

        def draw_rect(self, screen, top_left, bot_right):
            top_right = (bot_right[0], top_left[1])
            bot_left = (top_left[0], bot_right[1])

            pygame.draw.line(screen, GREEN, top_left, top_right, 3)
            pygame.draw.line(screen, GREEN, top_left, bot_left, 3)
            pygame.draw.line(screen, GREEN, bot_right, top_right, 3)
            pygame.draw.line(screen, GREEN, bot_left, bot_right, 3)

        def draw_word(self, screen, item):
            x = item[1][1] * TILE_SIZE + (item[1][1] + 1) * GAP_SIZE
            y = item[1][0] * TILE_SIZE + (item[1][0] + 1) * GAP_SIZE

            for i in range(len(item[0])):  
                if item[1][2] == 1:
                    draw_tile(screen, x, y + i * (TILE_SIZE + GAP_SIZE), item[0][i], board.VALUES, DARK_GREEN)
                else:
                    draw_tile(screen, x + i * (TILE_SIZE + GAP_SIZE), y, item[0][i], board.VALUES, DARK_GREEN)

        def erase_word(self, screen, item):
            row = item[1][0]
            col = item[1][1]
            x = col * TILE_SIZE + (col + 1) * GAP_SIZE
            y = row * TILE_SIZE + (row + 1) * GAP_SIZE

            for i in range(len(item[0])):
                if item[1][2] == 1:
                    put_tile(x, y + i * (TILE_SIZE + GAP_SIZE), row + i, col, board, screen)
                else:
                    put_tile(x + i * (TILE_SIZE + GAP_SIZE), y, row, col + i, board, screen)

        def draw(self, screen):
            list_rect = pygame.Rect(self.x, self.y - 6, self.width, self.height + 6)
            screen.fill(BURN, list_rect)

            visible_items = self.get_visible_items()
            for i, item in enumerate(visible_items):
                if item == self.selected:
                    top_left = (self.x + 1, self.y + i * self.item_height - 5)
                    bot_right = (self.x - 2 + self.width, self.y + (i + 1) * self.item_height - 5)
                    self.draw_rect(screen, top_left, bot_right)
                    self.erase_word(screen, self.prev)
                    self.draw_word(screen, item)
            
                pt_surface = self.render_item(item[2])  # Call your rendering function
                wd_surface = self.render_item(item[0])
                screen.blit(pt_surface, (self.x + 7, self.y + i * self.item_height))
                screen.blit(wd_surface, (self.x + 80, self.y + i * self.item_height))

            bot_rect = pygame.Rect(self.x, self.y + self.height, self.width, 50)
            screen.fill(DARK_BROWN, bot_rect)

        def render_item(self, item):
            if isinstance(item, int):
                text_surface = self.pts_font.render(str(item), True, CREAM)
            else:
                text_surface = self.word_font.render(item, True, CREAM)
            return text_surface

        def handle_scroll(self, delta):
            # Update scroll position based on user input (e.g., mouse wheel)
            self.scroll_pos = max(0, min(len(plays) - 1, self.scroll_pos - delta))  # Clamp scroll within data boundaries

        def handle_mouseover(self):
            pos = pygame.mouse.get_pos()
            if pos[0] > self.x and pos[0] < self.x + self.width and pos[1] > self.y and pos[1] < self.y + self.height:
                visible_items = self.get_visible_items()
                i = (pos[1] - self.y) // self.item_height
                if i < len(visible_items):
                    self.prev = self.selected
                    self.selected = visible_items[i]

    screen_width = 15 * TILE_SIZE + 16 * GAP_SIZE + 300
    screen_height = screen_width - 300
    
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Scrabble Board")

    screen.fill(GOLD)
    pygame.draw.rect(screen, DARK_BROWN, (screen_height, 0, 300, screen_height))
    for row in range(15):
        for col in range(15):
            x = col * TILE_SIZE + (col + 1) * GAP_SIZE
            y = row * TILE_SIZE + (row + 1) * GAP_SIZE
            put_tile(x, y, row, col, board, screen)

    my_list = ScrollableList(screen_height + 25, 60, 250, screen_height - 100, pygame.font.Font(None, 30))
    my_list.draw(screen)
    pts = my_list.render_item("PTS")
    word = my_list.render_item("TOP PLAYS")
    screen.blit(pts, (my_list.x + 5, my_list.y - 35))
    screen.blit(word, (my_list.x + 80, my_list.y - 35))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEWHEEL:
                my_list.handle_scroll(event.y) 
                my_list.draw(screen)
            if event.type == pygame.MOUSEMOTION:
                my_list.handle_mouseover()
                my_list.draw(screen)
        pygame.display.flip()

    pygame.display.quit()