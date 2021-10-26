(define (domain polycraft)
    (:requirements :typing :disjunctive-preconditions :fluents :time :negative-preconditions)
    (:types cell)
    (:predicates (isAccessible ?c - cell) (adjacent ?c1 - cell ?c2 - cell) (trader_{trader_id}_at ?c - cell))
    (:functions  (cell_type ?c - cell) (count_{item_type}) (selectedItem))

    (:action place_tree_tap
        :parameters (?at - cell ?near_to - cell)
        :precondition (and
            (isAccessible ?at)
            (isAccessible ?near_to)
            (adjacent ?at ?near_to)
            (cell_type ?at {BlockType.AIR.value})
            (cell_type ?near_to {BlockType.LOG.value})
            (>= (count_{ItemType.TREE_TAP.name}) 1)
        )
        :effect (and
            (decrease (count_{ItemType.TREE_TAP}) 1)
            (assign (cell_type ?to) {ItemType.TREE_TAP.value})
        )
    )

    ; COLLECT
    (:action collect_{item_type}
        :parameters (?c - cell)
        :precondition (and
            (isAccessible ?c)
            (cell_type ?c {block_type})
        )
        :effect (and
            (increase (count_{item_type}) 1)
        )
    )

    ; TELEPORT_TO AND BREAK
    (:action break_{block_type}
        :parameters (?c - cell)
        :precondition (and
            (isAccessible ?c)
            (cell_type ?c {block_type})
        )
        :effect (and
            (increase (count_{item_type}) {items_per_block})
            (cell_type ?c {BlockType.AIR.value})
        )
    )


    ; TELEPORT_TO AND BREAK for block types that require iron pickaxe
    (:action break_{block_type}
        :parameters (?c - cell)
        :precondition (and
            (isAccessible ?c)
            (cell_type ?c {block_type})
            (= (selectedItem) {ItemType.IRON_PICKAXE.value})
        )
        :effect (and
            (increase (count_{item_type}) {items_per_block})
            (cell_type ?c {BlockType.AIR})
        )
    )

    ; SELECT
    (:action select_{item_type.name}
        :precondition (and
            (>= (count_{item_type.name}) 1)
        )
        :effect (and
            (assign (selectedItem) {item_type.value})
        )
    )

    ; CRAFT With crafting table
    (:action craft_recipe_{recipe_idx}
        :parameters (?from - cell)
        :precondition (and
            (cell_type ?from {BlockType.CRAFTING_TABLE.value})
            (isAccessible ?from)
            (>= (count_{recipe_input}) {input_quantity})
        )
        :effect (and
            (increase (count_{recipe_output}) {output_quantity})
            (decrease (count_{recipe_input}) {input_quantity})
        )
    )

    ; CRAFT Without crafting table
    (:action craft_recipe_{recipe_idx}
        :precondition (and
            (>= (count_{recipe_input}) {input_quantity})
        )
        :effect (and
            (increase (count_{recipe_output}) {output_quantity})
            (decrease (count_{recipe_input}) {input_quantity})
        )
    )

    ; TRADE
    (:action trade_recipe_{trader_id}_{trade_idx}
        :parameters (?trader_loc - cell)
        :precondition (and
            (trader_{trader_id}_at ?trader_loc)
            (isAccessible ?trader_loc)
            (>= (count_{trade_input}) {input_quantity})
        )
        :effect (and
            (increase (count_{trade_output}) {output_quantity})
            (decrease (count_{trade_input}) {input_quantity})
        )
    )

    (:event cell_accessible
        :parameters (?c1 - cell ?c2 - cell)
        :precondition (and
            (isAccessible ?c1)
            (not (isAccessible ?c2))
            (cell_type ?c1 {BlockType.AIR})
            (adjacent ?c1 ?c2)
        )
        :effect (and
            (isAccessible ?c2)
        )
    )
)
