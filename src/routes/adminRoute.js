import express from "express";
import {
  createPlan,
  getPlans,
  getPlanById,
  updatePlan,
  deletePlan,
} from "../controllers/planController.js";


const router = express.Router();
router.get("/", getPlans);          
router.get("/:id", getPlanById);   

router.post("/", createPlan); 
router.put("/:id", updatePlan);   
router.delete("/:id", deletePlan);  

export default router;
