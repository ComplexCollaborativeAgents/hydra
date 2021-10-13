(define (domain polycraft)
    (:requirements :typing :durative-actions :disjunctive-preconditions :duration-inequalities :fluents :time :negative-preconditions :timed-initial-literals)
    (:types cell inventory_item)
    (:predicates (isAccessible ?c - cell) (open ?c - cell) (powered ?c - cell))
    (:functions (x ?c - cell) (y ?c - cell) (z ?c - cell) (steve_x) (steve_y) (steve_z) (steve_facing) (steve_yaw) (steve_pitch) (facing_block) (cell_type ?c - cell)
    (item_type ?i - inventory_item) (count ?i inventory_item) (crafted_item) (crafted_item_count) (traded_item) (traded_item_count) )

    (:action tp
        :parameters (?to - cell)
        :precondition (and
            (isAccessible ?to)
        )
        :effect (and
            (assign (steve_x) (x ?to) )
            (assign (steve_y) (y ?to) )
            (assign (steve_z) (z ?to) )
        )
    )

    (:action break_bedrock
        :parameters (?target - cell)
        :precondition (and
            (isAccessible ?target)
            (= (cell_type ?target)  1)
        )
        :effect (and
            (= (cell_type ?target)  0)
        )
    )
    (:action break_log
        :parameters (?target - cell)
        :precondition (and
            (isAccessible ?target)
            (= (cell_type ?target)  2)
        )
        :effect (and
            (= (cell_type ?target)  0)
        )
    )
    (:action break_diamond_ore
        :parameters (?target - cell)
        :precondition (and
            (isAccessible ?target)
            (= (cell_type ?target)  3)
        )
        :effect (and
            (= (cell_type ?target)  0)
        )
    )
    (:action break_plastic_chest
        :parameters (?target - cell)
        :precondition (and
            (isAccessible ?target)
            (= (cell_type ?target)  3)
        )
        :effect (and
            (= (cell_type ?target)  0)
        )
    )
    (:action break_wooden_door
        :parameters (?target - cell)
        :precondition (and
            (isAccessible ?target)
            (= (cell_type ?target)  7)
        )
        :effect (and
            (= (cell_type ?target)  0)
        )
    )
    (:action break_block_of_platinum
        :parameters (?target - cell)
        :precondition (and
            (isAccessible ?target)
            (= (cell_type ?target)  8)
        )
        :effect (and
            (= (cell_type ?target)  0)
        )
    )
)

