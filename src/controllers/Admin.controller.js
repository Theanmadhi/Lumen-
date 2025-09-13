import {Plan} from "../models/Plan.js";

export const createPlan = async (req, res) => {
  try {
    const plan = new Plan(req.body);
    await plan.save();
    res.status(201).json({ message: "Plan created successfully", data: plan });
  } catch (error) {
    res.status(400).json({message: error.message });
  }
};

export const getPlans = async (req, res) => {
  try {
    const { status } = req.query; // optional filter
    const query = status ? { status } : {};
    const plans = await Plan.find(query).sort({ price: 1 });
    res.status(200).json({data: plans });
  } catch (error) {
    res.status(500).json({message: error.message });
  }
};

export const getPlanById = async (req, res) => {
  try {
    const plan = await Plan.findById(req.params.id)
      .populate("downgradeTo", "name tier price")
      .populate("upgradeTo", "name tier price");

    if (!plan) {
      return res.status(404).json({ success: false, message: "Plan not found" });
    }

    res.status(200).json({data: plan });
  } catch (error) {
    res.status(500).json({message: error.message });
  }
};

//doubt
export const updatePlan = async (req, res) => {
  try {
    const plan = await Plan.findByIdAndUpdate(req.params.id, req.body, {
      new: true,
      runValidators: true,
    });

    if (!plan) {
      return res.status(404).json({message: "Plan not found" });
    }

    res.status(200).json({message: "Plan updated successfully", data: plan });
  } catch (error) {
    res.status(400).json({  message: error.message });
  }
};

export const deletePlan = async (req, res) => {
  try {
    const plan = await Plan.findByIdAndDelete(req.params.id);

    if (!plan) {
      return res.status(404).json({message: "Plan not found" });
    }

    res.status(200).json({message: "Plan deleted successfully" });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};