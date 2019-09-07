#pragma once

#include <cmath>

#include <SFML/Graphics.hpp>

#include "tree.h"

class GTree {
protected:
    sf::RenderWindow window;
    sf::Font font;
    const Tree tree;
    const float cell_size;

    void draw_vertex(Vertex *vertex, int level) {
        float vertex_size = cell_size * 0.7f;
        float x = (vertex->data + 1) * cell_size;
        float y = level * cell_size;
        sf::CircleShape shape(vertex_size);
        shape.setFillColor(sf::Color::White);
        shape.setOutlineThickness(1.f);
        shape.setOutlineColor(sf::Color::Black);
        shape.setPosition(x, y);
        window.draw(shape);
        sf::Text text;
        text.setFont(font);
        text.setString(std::to_string(vertex->data));
        text.setFillColor(sf::Color::Black);
        text.setPosition(x + vertex_size / 6.f, y + vertex_size / 7.f);
        text.setCharacterSize(unsigned(std::round(vertex_size * 1.4f)));
        window.draw(text);
    }


    virtual void draw_lines(Vertex *vertex, int level) {
        float to_center = cell_size * 0.6f;
        sf::Vector2f start((vertex->data + 1) * cell_size + to_center, level * cell_size + to_center);
        if (vertex->left) {
            sf::Vector2f finish((vertex->left->data + 1) * cell_size + to_center, (level + 2) * cell_size + to_center);
            sf::Vertex line[] = {start, finish};
            line[0].color = line[1].color = sf::Color::Black;
            window.draw(line, 2, sf::Lines);
        }
        if (vertex->right) {
            sf::Vector2f finish((vertex->right->data + 1) * cell_size + to_center, (level + 2) * cell_size + to_center);
            sf::Vertex line[] = {start, finish};
            line[0].color = line[1].color = sf::Color::Black;
            window.draw(line, 2, sf::Lines);
        }
    }

    virtual void draw_tree(Vertex *vertex, int level) {
        if (vertex) {
            draw_lines(vertex, level);
            draw_vertex(vertex, level);
            draw_tree(vertex->left, level + 2);
            draw_tree(vertex->right, level + 2);
        }
    }

public:
    explicit GTree(const Tree &tree, float scale = 1, unsigned width = 1366, unsigned height = 768) :
            tree(tree), cell_size(scale * 13.f) {
        sf::ContextSettings settings;
        settings.antialiasingLevel = 8;
        window.create(sf::VideoMode(width, height), "Tree", sf::Style::Default, settings);
        font.loadFromFile("/usr/share/fonts/noto/NotoSans-Regular.ttf");
    }

    virtual void start() {
        while (window.isOpen()) {
            sf::Event event;
            while (window.pollEvent(event)) {
                if (event.type == sf::Event::Closed)
                    window.close();
            }

            window.clear(sf::Color::White);
            draw_tree(tree.getRoot(), 1);
            window.display();
        }
    }
};

class GBinaryTree : public GTree {
public:
    explicit GBinaryTree(const Tree &tree, float scale = 1, unsigned width = 1366, unsigned height = 768) :
            GTree(tree, scale, width, height) {}

    void start() override {
        while (window.isOpen()) {
            sf::Event event;
            while (window.pollEvent(event)) {
                if (event.type == sf::Event::Closed)
                    window.close();
            }

            window.clear(sf::Color::White);
            draw_tree(tree.getRoot(), 1);
            window.display();
        }
    }

private:
    void draw_lines(Vertex *vertex, int level) override {
        float to_center = cell_size * 0.6f;
        sf::Vector2f start((vertex->data + 1) * cell_size + to_center, level * cell_size + to_center);
        if (vertex->left) {
            sf::Vector2f finish((vertex->left->data + 1) * cell_size + to_center, (level + 2) * cell_size + to_center);
            sf::Vertex line[] = {start, finish};
            line[0].color = line[1].color = sf::Color::Black;
            window.draw(line, 2, sf::Lines);
        }
        if (vertex->right && vertex->balance == 0) {
            sf::Vector2f finish((vertex->right->data + 1) * cell_size + to_center, (level + 2) * cell_size + to_center);
            sf::Vertex line[] = {start, finish};
            line[0].color = line[1].color = sf::Color::Black;
            window.draw(line, 2, sf::Lines);
        } else if (vertex->right && vertex->balance == 1) {
            sf::Vector2f finish((vertex->right->data + 1) * cell_size + to_center, (level) * cell_size + to_center);
            sf::Vertex line[] = {start, finish};
            line[0].color = line[1].color = sf::Color::Black;
            window.draw(line, 2, sf::Lines);

        }
    }

    void draw_tree(Vertex *vertex, int level) override {
        if (vertex && vertex->balance == 0) {
            draw_lines(vertex, level);
            draw_vertex(vertex, level);
            draw_tree(vertex->left, level + 2);
            draw_tree(vertex->right, level + 2);
        } else if (vertex && vertex->balance == 1) {
            draw_lines(vertex, level);
            draw_vertex(vertex, level);
            draw_tree(vertex->left, level + 2);
            draw_tree(vertex->right, level);
        }
    }
};

//class GDopTree : GTree {
//private:
//    const DopTree dopTree;
//public:
//    explicit GDopTree(const DopTree &tree, float scale = 1, unsigned width = 1366, unsigned height = 768) :
//            GTree(tree, scale, width, height), dopTree(tree) {}
//private:
//    void draw_weights(Vertex *vertex, int level, const int arr[100], int &n) {
//        if (vertex) {
//            draw_weights(vertex->left, level + 2, arr, n);
//            float x = (vertex->data + 1) * cell_size;
//            float y = level * cell_size;
//            sf::Text text;
//            text.setFont(font);
//            text.setString(std::to_string(arr[n++]));
//            text.setFillColor(sf::Color::Black);
//            text.setPosition(x - 5, y - 5);
//            text.setCharacterSize(unsigned(std::round(cell_size/1.6f)));
//            window.draw(text);
//            draw_weights(vertex->right, level + 2, arr, n);
//        }
//    }
//public:
//    void start() override {
//        while (window.isOpen()) {
//            sf::Event event;
//            while (window.pollEvent(event)) {
//                if (event.type == sf::Event::Closed)
//                    window.close();
//            }
//
//            window.clear(sf::Color::White);
//            draw_tree(tree.getRoot(), 1);
//            int i = 0;
//            draw_weights(tree.getRoot(), 1, dopTree.w, i);
//            window.display();
//        }
//    }
//};