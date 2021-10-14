(define (domain polycraft)
    (:requirements :typing :disjunctive-preconditions :fluents :time :negative-preconditions)
    (:types cell inventory_item entity)
    (:predicates (isAccessible ?c - cell) (open ?c - cell) (powered ?c - cell) (pogo_created))
    (:functions (cell_x ?c - cell) (cell_y ?c - cell) (cell_z ?c - cell) (entity_x ?e - entity) (entity_y ?e - entity) (entity_z ?e - entity) (steve_x) (steve_y) (steve_z) (steve_facing) (steve_yaw) (steve_pitch) (facing_block) (cell_type ?c - cell) (id ?e - entity)
    (item_type ?i - inventory_item) (count ?i - inventory_item) (crafted_item) (crafted_item_count) (traded_item) (traded_item_count) )

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

