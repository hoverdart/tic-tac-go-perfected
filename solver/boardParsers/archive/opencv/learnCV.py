
"""learnCV.py

Introductory OpenCV examples and a scaffold for OpenCVBoardParser.
This file demonstrates common steps useful when building and refining
an OpenCVBoardParser for detecting board games like Tic-Tac-Toe.

Requirements: opencv-python, numpy

Run: python learnCV.py
"""

import cv2
import numpy as np
from typing import List, Tuple


class OpenCVBoardParser:
	"""A scaffolded parser showing steps to detect a rectangular board,
	extract a top-down (bird's-eye) view, split into cells, and do basic
	symbol detection per cell.

	Implementations here are intentionally simple so you can extend & refine.
	"""

	def __init__(self, debug: bool = False):
		self.debug = debug

	def preprocess(self, image: np.ndarray) -> np.ndarray:
		"""Convert to grayscale and blur to reduce noise."""
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray, (5, 5), 0)
		return blur

	def find_edges(self, image: np.ndarray) -> np.ndarray:
		"""Canny edge detection."""
		edges = cv2.Canny(image, 50, 150)
		return edges

	def find_largest_quad(self, edges: np.ndarray) -> Tuple[np.ndarray, bool]:
		"""Find the largest 4-point contour (likely the board).

		Returns (approx_contour, found)
		"""
		contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		max_area = 0
		best = None
		for cnt in contours:
			area = cv2.contourArea(cnt)
			if area < 1000:
				continue
			peri = cv2.arcLength(cnt, True)
			approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
			if len(approx) == 4 and area > max_area:
				max_area = area
				best = approx
		return best, best is not None

	def order_points(self, pts: np.ndarray) -> np.ndarray:
		"""Order rectangle points: tl, tr, br, bl."""
		pts = pts.reshape(4, 2)
		s = pts.sum(axis=1)
		diff = np.diff(pts, axis=1)
		tl = pts[np.argmin(s)]
		br = pts[np.argmax(s)]
		tr = pts[np.argmin(diff)]
		bl = pts[np.argmax(diff)]
		return np.array([tl, tr, br, bl], dtype="float32")

	def four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
		rect = self.order_points(pts)
		(tl, tr, br, bl) = rect
		widthA = np.linalg.norm(br - bl)
		widthB = np.linalg.norm(tr - tl)
		maxWidth = int(max(widthA, widthB))
		heightA = np.linalg.norm(tr - br)
		heightB = np.linalg.norm(tl - bl)
		maxHeight = int(max(heightA, heightB))
		dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
		M = cv2.getPerspectiveTransform(rect, dst)
		warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
		return warped

	def split_grid(self, board_img: np.ndarray, rows: int, cols: int) -> List[np.ndarray]:
		h, w = board_img.shape[:2]
		cell_h = h // rows
		cell_w = w // cols
		cells = []
		for r in range(rows):
			for c in range(cols):
				y1 = r * cell_h
				x1 = c * cell_w
				cell = board_img[y1:y1 + cell_h, x1:x1 + cell_w]
				cells.append(cell)
		return cells

	def detect_symbol_in_cell(self, cell: np.ndarray) -> str:
		"""Very simple symbol detection: detect if there's a circle (O) or
		significant lines (X). Returns 'O', 'X', or '' for empty/unknown.
		"""
		gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
		thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
									   cv2.THRESH_BINARY_INV, 11, 3)
		# Circle detection
		circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
								   param1=50, param2=25, minRadius=5, maxRadius=100)
		if circles is not None:
			return 'O'
		# Line detection (rough for X)
		edges = cv2.Canny(thresh, 50, 150)
		lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=10, maxLineGap=10)
		if lines is not None and len(lines) >= 2:
			return 'X'
		return ''

	def parse_board(self, image: np.ndarray, rows: int = 3, cols: int = 3) -> Tuple[List[str], np.ndarray]:
		"""High level pipeline: returns list of symbols row-major and the warped board image."""
		pre = self.preprocess(image)
		edges = self.find_edges(pre)
		quad, found = self.find_largest_quad(edges)
		if not found:
			if self.debug:
				print("No board found")
			return [], image
		warped = self.four_point_transform(image, quad)
		cells = self.split_grid(warped, rows, cols)
		symbols = [self.detect_symbol_in_cell(c) for c in cells]
		return symbols, warped


def demo_from_camera():
	cap = cv2.VideoCapture(0)
	parser = OpenCVBoardParser(debug=True)
	if not cap.isOpened():
		print("Cannot open camera")
		return
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		symbols, warped = parser.parse_board(frame)
		# simple visualization
		out = frame.copy()
		if warped is not None:
			cv2.imshow('Warped Board', warped)
		if symbols:
			print('Symbols:', symbols)
		cv2.imshow('Camera', out)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	# Quick demo: attempt to use the webcam. Press 'q' to quit.
	demo_from_camera()
