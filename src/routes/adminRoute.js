import { Router } from "express";
import { createPlan, getPlans, getPlanById } from "../controllers/Admin.controller.js";

const router = Router();

router.route("/plans").post(createPlan).get(getPlans);
router.route("/plans/:id").get(getPlanById);

export default router;
