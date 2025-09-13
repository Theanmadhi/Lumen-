import jwt from "jsonwebtoken"
import {User} from "../models/User.model.js"
import httpStatus from "http-status"

const protectRoute = async (req, res, next) => {
  const token = req.headers["authorization"]?.split(" ")[1];
  if (!token) return res.status(httpStatus.UNAUTHORIZED).json({ error: "No token provided" });

  try {
    const decoded = jwt.verify(token, "mysecretkey");
    const user = await User.findById(decoded.id);

    if (!user) return res.status(401).json({ error: "User not found" });
    if(!user.isVerified) return res.status(httpStatus.UNAUTHORIZED).json({error:"User needs to verify the email"});

    req.user = { id: user._id.toString()};
    next();

  } catch (err) {
    res.status(httpStatus.UNAUTHORIZED).json({ error: "Invalid token" });
  }
};

export {protectRoute};
